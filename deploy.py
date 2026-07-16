"""Modal deploy entry for SAM-Audio (Meta, text-prompted sound separation).

Implements three ABI slots with one model:
  - ``denoise_audio``         -> isolate the primary content (default: speech)
  - ``separate_audio_track``  -> isolate the vocals from a mix
  - ``music-extract``         -> isolate any sound described by ``track`` text

SAM-Audio separates any sound from a mixture given a natural-language
description; the ``track`` field of music-extract is passed through verbatim,
so it accepts free text ("dog barking", "the piano in the background"), not
just fixed stem names.

The ``facebook/sam-audio-*`` checkpoints are gated on Hugging Face: accept the
terms on the repo page, then put that account's ``HF_TOKEN`` in TongFlow
Settings before first use.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import modal
from tongflow import deploy
from tongflow.models.denoise_audio import DenoiseAudioInput, DenoiseAudioOutput
from tongflow.models.music_extract import MusicExtractInput, MusicExtractOutput
from tongflow.models.separate_audio_track import (
    SeparateAudioTrackInput,
    SeparateAudioTrackOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import node_slot

REPO_URL = "https://github.com/facebookresearch/sam-audio.git"
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = "bb4c6999d2677c7402360e426afc01ddfad6dce0"

# Plugin-internal knobs — NOT ABI fields. The denoise/vocals prompts are what
# SAM-Audio is asked to keep; reranking and span prediction trade latency for
# quality (README: both have a large impact on results).
MODEL_ID = os.environ.get("SAM_AUDIO_MODEL", "facebook/sam-audio-large")
DENOISE_PROMPT = os.environ.get("SAM_AUDIO_DENOISE_PROMPT", "clear speech")
VOCALS_PROMPT = os.environ.get("SAM_AUDIO_VOCALS_PROMPT", "singing vocals")
RERANK = int(os.environ.get("SAM_AUDIO_RERANK", 4))
PREDICT_SPANS = os.environ.get("SAM_AUDIO_PREDICT_SPANS", "1") not in ("0", "false")
# RoPE positions cap the usable context at ~400 s of 48 kHz audio; refuse
# inputs beyond this rather than produce garbage.
MAX_SECONDS = float(os.environ.get("SAM_AUDIO_MAX_SECONDS", 300))

volume = modal.Volume.from_name("models", create_if_missing=True)
# The checkpoint repo is gated: forward the local HF_TOKEN (TongFlow Settings)
# into the container so snapshot downloads authenticate.
secrets = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # ffmpeg: torchcodec runtime + our own decode-to-wav fast path;
    # libsndfile1: soundfile read/write.
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("soundfile", "numpy")
    .pip_install(f"git+{REPO_URL}@{REPO_REV}")
    .pip_install("tongflow==0.2.6")
    # All HF downloads (checkpoint, T5, judge, CLAP, PE span predictor) cache
    # on the shared volume so cold starts after the first are download-free.
    .env({"HF_HOME": "/models/hf"})
)

with image.imports():
    import io
    import subprocess
    import tempfile

    import soundfile as sf
    import torch


@deploy
@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    secrets=[secrets],
    timeout=1800,
    scaledown_window=5,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Load SAM-Audio once; reused across calls (warm).

        ``visual_ranker=None`` drops the ImageBind reranker — it is only used
        for visual prompts, which this plugin never sends — saving its
        multi-GB download and VRAM. The text ranker stays: it picks the best
        of ``RERANK`` candidates.
        """
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.model = (
            SAMAudio.from_pretrained(MODEL_ID, visual_ranker=None).eval().cuda()
        )
        self.processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
        # Persist whatever the first boot downloaded (CLAP, span predictor)
        # so later cold starts are download-free.
        volume.commit()

    @staticmethod
    def _as_wav(stack: contextlib.ExitStack, media: Any) -> str:
        """Materialize an incoming audio Asset as a mono 48 kHz WAV file.

        The processor accepts any-SR files, but only formats libsndfile /
        torchcodec can read; transcoding through ffmpeg first makes every
        container (m4a/aac/webm/...) work and lets us cheaply check duration.
        """
        src = str(stack.enter_context(asset_as_path(media)))
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        stack.callback(lambda: os.path.exists(wav) and os.unlink(wav))
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "48000", wav],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-300:]
            raise RuntimeError(f"could not decode audio input: {tail}")
        info = sf.info(wav)
        if info.frames / info.samplerate > MAX_SECONDS:
            raise RuntimeError(
                f"audio is {info.frames / info.samplerate:.0f}s long; "
                f"SAM-Audio supports at most {MAX_SECONDS:.0f}s per call"
            )
        return wav

    def _separate(self, media: Any, description: str) -> bytes:
        """Isolate ``description`` from the mixture; return WAV bytes."""
        with contextlib.ExitStack() as stack:
            wav = self._as_wav(stack, media)
            batch = self.processor(audios=[wav], descriptions=[description]).to(
                "cuda"
            )
            with torch.inference_mode():
                result = self.model.separate(
                    batch,
                    predict_spans=PREDICT_SPANS,
                    reranking_candidates=RERANK,
                )
        target = result.target[0].cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, target, self.model.sample_rate, format="WAV")
        return buf.getvalue()

    @modal.method()
    @node_slot(NodeSlots.DENOISE_AUDIO)
    def denoise_audio(self, input: DenoiseAudioInput) -> DenoiseAudioOutput:
        try:
            raw = self._separate(input.fileKey, DENOISE_PROMPT)
        except Exception as e:
            return DenoiseAudioOutput(success=False, error=str(e))
        return DenoiseAudioOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.SEPARATE_AUDIO_TRACK)
    def separate_audio_track(
        self, input: SeparateAudioTrackInput
    ) -> SeparateAudioTrackOutput:
        try:
            raw = self._separate(input.audio, VOCALS_PROMPT)
        except Exception as e:
            return SeparateAudioTrackOutput(success=False, error=str(e))
        return SeparateAudioTrackOutput(
            success=True,
            file_key=asset(raw, mime="audio/wav", filename="separated_vocals.wav"),
        )

    @modal.method()
    @node_slot(NodeSlots.MUSIC_EXTRACT)
    def music_extract(self, input: MusicExtractInput) -> MusicExtractOutput:
        try:
            track = (input.track or "").strip()
            if not track:
                raise RuntimeError("track description is required")
            if input.seed is not None:
                torch.manual_seed(int(input.seed))
            raw = self._separate(input.audio, track)
        except Exception as e:
            return MusicExtractOutput(success=False, error=str(e))
        return MusicExtractOutput(success=True, audio=asset(raw, mime="audio/wav"))
