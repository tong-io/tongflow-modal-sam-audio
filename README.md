# tongflow-modal-sam-audio

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Text-prompted sound separation with **SAM-Audio** (Meta, `facebook/sam-audio-large`), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Noise reduction** (`denoise_audio`) — isolate the clean speech from a noisy recording.
- **Vocal separation** (`separate_audio_track`) — pull the vocals out of a mix.
- **Extract track** (`music-extract`) — isolate *any* sound described in free text ("dog barking", "the piano in the background", "drums"), not just fixed stem names.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | The checkpoint is gated: request access on [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large), then use that account's token. |

On first use the plugin deploys to your Modal account automatically and caches the build; weights are cached on a shared Modal volume.

## Tuning (env, optional)

| Env | Default | Notes |
| --- | --- | --- |
| `SAM_AUDIO_MODEL` | `facebook/sam-audio-large` | Any `facebook/sam-audio-*` variant. |
| `SAM_AUDIO_RERANK` | `4` | Candidates re-ranked per call; higher = better/slower. |
| `SAM_AUDIO_PREDICT_SPANS` | `1` | Predict where the target sound occurs first. |
| `SAM_AUDIO_DENOISE_PROMPT` | `clear speech` | What `denoise_audio` keeps. |
| `SAM_AUDIO_VOCALS_PROMPT` | `singing vocals` | What `separate_audio_track` keeps. |
| `SAM_AUDIO_MAX_SECONDS` | `300` | Longest accepted input. |
