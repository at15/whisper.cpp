# Windows Issues

On windows, besides speed, the chrome extension is also less responsive and buggy

- closing the popup, the offscreen document is still running, reopen the popup, it does not show recording
- cannot ctrl c the server unless I disable the chrome extension

The performance is really bad compared with mac on windows.

- VAD its taking 560ms
- Infer is taking even longer e.g. 3189ms

I think it might be using single thread...
We can either consider using cuda or openblas.
Both VAD and whsiper.cpp should be able to use CUDA.
Or we can try using OpenBLAS.

```
[20:22:37.891] INFER session=fa1821d8-466f-4421-b922-84ec694e15e3 DONE time=1550ms text_len=72 text="Φ▒åΦàÉΣ╣│µÿ»µêæΣ╗¼Φ┐ÖΣ╕¬Θà▒τë¢ΦéëτÜäτ¼¼Σ╕ëΘà▒Σ╕║Σ╗..."
[20:22:37.892] PUSH session=fa1821d8-466f-4421-b922-84ec694e15e3 chunk=378 audio=504ms total=191016ms buffer=504ms
whisper_vad_detect_speech: detecting speech in 8064 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 384 < n_window: 512
whisper_vad_detect_speech: vad time = 562.14 ms processing 8064 samples
[20:22:37.893] SEGMENT session=fa1821d8-466f-4421-b922-84ec694e15e3 index=33 t0=18597 t1=19101 text="Φ▒åΦàÉΣ╣│µÿ»µêæΣ╗¼Φ┐ÖΣ╕¬Θà▒τë¢ΦéëτÜäτ¼¼Σ╕ëΘà▒Σ╕║Σ╗ÇΣ╣êΦ┐ÖΣ╣êΦ»┤σæóσëìΘ¥ó"
[20:22:37.893] INFER session=fa1821d8-466f-4421-b922-84ec694e15e3 iter=0 audio=504ms pending=0ms prob=1.000
[20:22:41.083] INFER session=fa1821d8-466f-4421-b922-84ec694e15e3 DONE time=3189ms text_len=33 text="µêæτÅ╛σ£¿ΦªüΣ╛åτ£ïτ£ïΣ╜áτÜäΦçëΦë▓"
[20:22:41.083] PUSH session=fa1821d8-466f-4421-b922-84ec694e15e3 chunk=379 audio=504ms total=191520ms buffer=504ms
whisper_vad_detect_speech: detecting speech in 8064 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 384 < n_window: 512
whisper_vad_detect_speech: vad time = 563.09 ms processing 8064 samples
[20:22:41.085] PUSH session=fa1821d8-466f-4421-b922-84ec694e15e3 chunk=380 audio=504ms total=192024ms buffer=504ms
whisper_vad_detect_speech: detecting speech in 8064 samples
whisper_vad_detect_speech: n_chunks: 16
whisper_vad_detect_speech: props size: 16
whisper_vad_detect_speech: chunk_len: 384 < n_window: 512
whisper_vad_detect_speech: vad time = 564.01 ms processing 8064 samples
```