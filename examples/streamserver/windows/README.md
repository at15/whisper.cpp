# Windows

This folder contains doc on various issues we are facing on windows.

- [Build](build.md) how to build and run the server on windows
- [Speed](speed.md) slow VAD and transcribe on windows
- [UI](ui.md) issues with the chrome extension on windows

## Usage

```powershell
# downlod the models
# TODO: not downloaded to models folder like mac
.\models\download-ggml-model.cmd base
.\models\download-ggml-model.cmd medium
.\models\download-vad-model.cmd silero-v6.2.0

# run the server with vad
.\build\bin\Release\whisper-streamserver `
    -m ggml-base.bin `
    --vad-model ggml-silero-v6.2.0.bin `
    --vad-threshold 0.5 `
    --vad-min-speech 250 `
    --vad-min-silence 100 `
    --vad-max-speech 5000 `
    --inference-interval 1000 `
    --step 500 `
    --host 127.0.0.1 `
    --port 8080
```