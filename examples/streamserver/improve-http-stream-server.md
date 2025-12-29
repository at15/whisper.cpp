# Improve HTTP Stream Server

## Background

Right now we don't have any test for our http stream server.
I updated the chrome extension to record some audio.
I want to focus on transcribe for now because translate
can only translate to english and often lead to garbage like
`I am not good at English`.

We should take a deeper look at the VAD logic.
Seems `whisper-cli` also supports VAD and it worked better
than our stream server for the Chinese sample audio I added.

```bash
./build/bin/whisper-cli \
    --file samples/recording-zh-xiaogao-jie.wav \
    --model models/ggml-medium.bin \
    --language zh \
    --vad \
    --vad-model models/ggml-silero-v6.2.0.bin \
    --output-srt
```

I want to you to do the following:

- Create a cli (could reuse same binary of server) to make our testing easier
  - the cli should be able to read from local audio file and simulate a streaming input.
  - Output the result with timestamp so we can compare it with `whisper-cli`'s output
- Improve the stream server's logic by iterating on the cli, the cli and server should share the same logic