# Building whisper.cpp on Windows

## Prerequisites

- **Visual Studio 2019 or later** (with C++ desktop development workload)
  - Download from: https://visualstudio.microsoft.com/downloads/
  - During installation, select "Desktop development with C++" workload
- **CMake 3.5 or later**
  - Download from: https://cmake.org/download/
  - Or install via Visual Studio Installer (included in C++ workload)
- **Git**
  - Download from: https://git-scm.com/download/win

## Building with MSVC (Visual Studio)

### Step 1: Clone the repository

```powershell
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

### Step 2: Configure with CMake

For 64-bit build (recommended):

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release
```

For 32-bit build:

```powershell
cmake -S . -B build -A Win32 -DCMAKE_BUILD_TYPE=Release
```

### Step 3: Build the project

```powershell
cmake --build build --config Release -j
```

Or open the solution file in Visual Studio:

```powershell
start build\whisper.cpp.sln
```

Then build from Visual Studio (Build → Build Solution or press `Ctrl+Shift+B`).

### Step 4: Download a model

```powershell
.\models\download-ggml-model.cmd base.en
```

Available models: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`

### Step 5: Run whisper-cli

```powershell
.\build\bin\Release\whisper-cli.exe -m models\ggml-base.en.bin -f samples\jfk.wav
```

## Building with SDL2 support (for real-time audio)

If you want to build examples that use SDL2 (like `whisper-stream`):

### Step 1: Download SDL2

Download SDL2 development libraries from: https://github.com/libsdl-org/SDL/releases

Extract to a location like `C:\SDL2-2.28.5\`

### Step 2: Set SDL2_DIR environment variable

```powershell
$env:SDL2_DIR = "C:\SDL2-2.28.5\cmake"
```

### Step 3: Configure with SDL2 enabled

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DWHISPER_SDL2=ON
```

### Step 4: Build

```powershell
cmake --build build --config Release -j
```

### Step 5: Copy SDL2.dll to output directory

```powershell
Copy-Item "C:\SDL2-2.28.5\lib\x64\SDL2.dll" -Destination "build\bin\Release\"
```

## Building with OpenBLAS support (optional, for faster CPU inference)

### Step 1: Download OpenBLAS

Download from: https://github.com/OpenMathLib/OpenBLAS/releases

Extract to a location like `C:\OpenBLAS-0.3.23\`

### Step 2: Configure with OpenBLAS

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBLAS_LIBRARIES="C:\OpenBLAS-0.3.23\lib\libopenblas.lib" -DBLAS_INCLUDE_DIRS="C:\OpenBLAS-0.3.23\include"
```

### Step 3: Build

```powershell
cmake --build build --config Release -j
```

### Step 4: Copy OpenBLAS DLL

```powershell
Copy-Item "C:\OpenBLAS-0.3.23\bin\libopenblas.dll" -Destination "build\bin\Release\"
```

## Building with CUDA support (for NVIDIA GPUs)

### Step 1: Install CUDA Toolkit

Download and install from: https://developer.nvidia.com/cuda-downloads

### Step 2: Configure with CUDA

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=1
```

### Step 3: Build

```powershell
cmake --build build --config Release -j
```

## Building with Vulkan support (for GPU acceleration)

### Step 1: Install Vulkan SDK

Download and install from: https://vulkan.lunarg.com/sdk/home

### Step 2: Configure with Vulkan

```powershell
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=1
```

### Step 3: Build

```powershell
cmake --build build --config Release -j
```

## Output locations

After building, executables will be located in:

- **Release builds**: `build\bin\Release\`
- **Debug builds**: `build\bin\Debug\`

Common executables:
- `whisper-cli.exe` - Command-line transcription tool
- `whisper-stream.exe` - Real-time audio streaming (requires SDL2)
- `whisper-streamserver.exe` - HTTP streaming server

## Troubleshooting

### CMake not found

Make sure CMake is in your PATH, or use the full path:

```powershell
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build -A x64
```

### Visual Studio not detected

Open "Developer Command Prompt for VS" or "Developer PowerShell for VS" from the Start menu, then run the build commands.

Alternatively, specify the generator explicitly:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
```

### Missing DLL errors

If you get DLL errors when running executables, make sure:
- SDL2.dll is copied to the output directory (if using SDL2)
- OpenBLAS DLL is copied (if using OpenBLAS)
- CUDA DLLs are in PATH (if using CUDA)
- Vulkan DLLs are in PATH (if using Vulkan)

## Alternative: Building with MinGW/MSYS2

If you prefer MinGW over MSVC:

### Step 1: Install MSYS2

Download from: https://www.msys2.org/

### Step 2: Open MSYS2 terminal (UCRT64 or CLANG64)

### Step 3: Install dependencies

```bash
pacman -S base-devel git mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake
```

### Step 4: Build

```bash
cmake -B build
cmake --build build --config Release -j
```

Executables will be in `build\bin\` directory.

