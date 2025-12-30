# Speed issues on windows

Using the longer audio sample

The VAD is slow for both, but the infer is 10 times slower on windows.

## Try official releases

Using the official release from https://github.com/ggml-org/whisper.cpp/releases

```powershell
.\build\bin\Release\whisper-cli.exe `
    --file samples\recording-zh-xiaogao-jie.wav `
    --model models\ggml-medium.bin `
    --language zh `
    --vad `
    --vad-model models\ggml-silero-v6.2.0.bin `
    --output-srt
```

```bash
./build/bin/whisper-cli \
    --file samples/recording-zh-xiaogao-jie.wav \
    --model models/ggml-medium.bin \
    --language zh \
    --vad \
    --vad-model models/ggml-silero-v6.2.0.bin \
    --output-srt
```

## Windows speed

From [windows-issues.md](windows-issues.md), we have the following logs:

- `whisper_vad_detect_speech: vad time = 563.09 ms processing 8064 samples`
- `INFER session=fa1821d8-466f-4421-b922-84ec694e15e3 DONE time=3189ms text_len=33`

## Mac speed

on mac, infer runs on metal and va runs on CPU (I suppose?)

- `INFER session=cli DONE time=280ms text_len=16 text="白胡椒,味精"`

```bash
./build/bin/whisper-streamserver \
    -m models/ggml-medium.bin \
    --vad-model models/ggml-silero-v6.2.0.bin \
    -f samples/recording-zh-xiaogao-jie.wav \
    -l zh \
    --output-srt \
    --vad-debug
```

```text
./build/bin/whisper-streamserver -m models/ggml-medium.bin --vad-model  -f  -  4.68s user 1.21s system 17% cpu 32.843 total
```

```text
21:22:14.283] VAD segment: CONTINUING (prob=1.000, pending=500ms)
[21:22:14.283] CLI chunk=94 samples=8000 total_ms=47500 (47.5s/49.4s)
whisper_vad_detect_speech: detecting speech in 8000 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 320 < n_window: 512
whisper_vad_detect_speech: vad time = 211.14 ms processing 8000 samples
[21:22:14.285] VAD segment: CONTINUING (prob=1.000, pending=1000ms)
[21:22:14.285] INFER session=cli iter=1 audio=1500ms pending=1000ms prob=1.000
[21:22:14.565] INFER session=cli DONE time=280ms text_len=16 text="白胡椒,味精"
[21:22:14.565] CLI chunk=95 samples=8000 total_ms=48000 (48.0s/49.4s)
whisper_vad_detect_speech: detecting speech in 8000 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 320 < n_window: 512
whisper_vad_detect_speech: vad time = 213.08 ms processing 8000 samples
[21:22:14.567] VAD segment: CONTINUING (prob=1.000, pending=1500ms)
[21:22:14.567] INFER session=cli iter=2 audio=2000ms pending=1500ms prob=1.000
[21:22:14.854] INFER session=cli DONE time=286ms text_len=25 text="白胡椒,味精一点点"
[21:22:14.854] CLI chunk=96 samples=8000 total_ms=48500 (48.5s/49.4s)
whisper_vad_detect_speech: detecting speech in 8000 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 320 < n_window: 512
whisper_vad_detect_speech: vad time = 214.96 ms processing 8000 samples
[21:22:14.856] VAD segment: CONTINUING (prob=1.000, pending=2000ms)
[21:22:14.856] INFER session=cli iter=3 audio=2500ms pending=2000ms prob=1.000
[21:22:15.159] INFER session=cli DONE time=303ms text_len=28 text="白胡椒,味精一点点13C"
[21:22:15.159] CLI chunk=97 samples=8000 total_ms=49000 (49.0s/49.4s)
whisper_vad_detect_speech: detecting speech in 8000 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 320 < n_window: 512
whisper_vad_detect_speech: vad time = 217.00 ms processing 8000 samples
[21:22:15.161] VAD segment: CONTINUING (prob=1.000, pending=2500ms)
[21:22:15.161] INFER session=cli iter=4 audio=3000ms pending=2500ms prob=1.000
[21:22:15.471] INFER session=cli DONE time=309ms text_len=34 text="白胡椒,味精一点点十三香"
[21:22:15.471] CLI chunk=98 samples=6272 total_ms=49392 (49.4s/49.4s)
[21:22:15.471] CLI final segment: index=11 t0=4650 t1=4939 text="白胡椒,味精一点点十三香"
[21:22:15.471] CLI completed: 12 segments
```