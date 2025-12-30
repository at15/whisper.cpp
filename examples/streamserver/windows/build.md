---
tags:
  - windows
  - vs
---

# Building whisper.cpp streamserver on Windows

## Status

I am using Visual Studio 2026, the CI is using Visual Studio 2022

- Likely need to downgrade to visual studio 2022 have CUDA (and unreal engine, working properly)
- https://aka.ms/vs/17/release/vs_community.exe from https://www.reddit.com/r/VisualStudio/comments/1p434d8/how_can_i_download_the_visual_studio_2022/ I guess you can install both of them

## Commands

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

If you already opened the folder using visual studio, you can build all.
From top menu, `Build`


Download model

```powershell
.\models\download-ggml-model.cmd base
```

Run on the sample

```powershell
.\build\bin\Release\whisper-cli.exe -m ggml-base.bin -f samples\jfk.wav
```
