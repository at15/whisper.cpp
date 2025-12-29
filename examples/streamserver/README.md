# Stream Server with HTTP Polling

HTTP-based streaming transcription server with VAD (Voice Activity Detection) support for live captioning.

## Background

This server enables live captioning of audio streams captured from browser tabs.
Unlike the standard HTTP server which requires complete audio files, this server:

- Accepts audio chunks via HTTP polling (no WebSocket required)
- Uses VAD to detect speech and avoid transcribing silence
- Maintains session state for continuous streaming
- Supports multiple concurrent sessions (processed sequentially)

## Building

```bash
cmake -B build
cmake --build build --config Release
```

The binary will be at `build/bin/whisper-streamserver`.

## Usage

```bash
./build/bin/whisper-streamserver -m models/ggml-base.bin

# Disable VAD completely
./build/bin/whisper-streamserver -m models/ggml-base.bin --no-vad

# Debug VAD to see energy values
./build/bin/whisper-streamserver -m models/ggml-base.bin --vad-debug

# Adjust threshold (default is 0.01, try lower values like 0.001)
./build/bin/whisper-streamserver -m models/ggml-base.bin --vad-thold 0.001
```

### Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t N, --threads N` | 4 | Number of threads |
| `-m FNAME, --model FNAME` | models/ggml-base.en.bin | Model path |
| `--step N` | 500 | Process every N ms of new audio |
| `--keep N` | 200 | Overlap from previous chunk in ms |
| `--length N` | 10000 | Max audio chunk for inference in ms |
| `--vad-thold N` | 0.01 | VAD energy threshold |
| `--freq-thold N` | 100.0 | High-pass filter cutoff Hz |
| `--host HOST` | 127.0.0.1 | Hostname |
| `--port PORT` | 8080 | Port |
| `--timeout N` | 60 | Session timeout in seconds |
| `-ng, --no-gpu` | - | Disable GPU |

## API Reference

### POST /stream/create

Create a new streaming session.

**Request** (JSON):
```json
{
  "session_id": "my-session-123",
  "language": "en",
  "translate": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| session_id | string | yes | - | Unique session identifier (client-provided) |
| language | string | no | "auto" | Source language code (e.g., "en", "zh", "ja") or "auto" |
| translate | bool | no | false | Translate to English |

**Response** (201 Created):
```json
{
  "status": "created",
  "session_id": "my-session-123",
  "language": "en",
  "translate": false
}
```

**Error** (409 Conflict):
```json
{
  "status": "error",
  "message": "session already exists"
}
```

### POST /stream/push

Upload audio chunk to session.

**Request** (multipart/form-data):
- `session_id`: string (required)
- `chunk_id`: int (required, for ordering)
- `file`: WAV audio data (16kHz mono)

**Response** (200 OK):
```json
{
  "status": "ok",
  "session_id": "my-session-123",
  "chunk_id": 5,
  "buffered_ms": 3500
}
```

**Error** (404 Not Found):
```json
{
  "status": "error",
  "message": "session not found or expired"
}
```

### GET /stream/poll

Retrieve transcription results.

**Query Parameters**:
- `session_id`: string (required)
- `after_index`: int (optional, only return segments after this index)

**Response** (200 OK):
```json
{
  "session_id": "my-session-123",
  "stream_ms": 15000,
  "segments": [
    {"index": 0, "text": "Hello world", "t0": 0, "t1": 250},
    {"index": 1, "text": "How are you", "t0": 260, "t1": 480}
  ],
  "current": {
    "text": "I am doing...",
    "t0": 490
  },
  "is_speaking": true
}
```

- `segments`: Finalized transcript segments
- `current`: In-progress partial transcription (may change)
- `is_speaking`: Whether speech is currently detected
- `t0`, `t1`: Timestamps in centiseconds (divide by 100 for seconds)

### DELETE /stream/{session_id}

Close session and get final results.

**Response** (200 OK):
```json
{
  "session_id": "my-session-123",
  "stream_ms": 20000,
  "segments": [...],
  "is_final": true
}
```

### GET /stream/health

Health check endpoint.

**Response** (200 OK):
```json
{
  "status": "ready",
  "active_sessions": 2
}
```

## Testing with curl

```bash
# Start server
./build/bin/whisper-streamserver -m models/ggml-base.en.bin

# Create session
curl -X POST http://localhost:8080/stream/create \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "language": "en"}'

# Push audio chunk
curl -X POST http://localhost:8080/stream/push \
  -F "session_id=test123" \
  -F "chunk_id=0" \
  -F "file=@chunk0.wav"

# Poll for results
curl "http://localhost:8080/stream/poll?session_id=test123"

# Close session
curl -X DELETE http://localhost:8080/stream/test123
```

## Chrome Extension Integration

To integrate with a Chrome extension for tab audio capture:

1. Generate a unique session_id (UUID) when starting capture
2. POST to `/stream/create` with session configuration
3. Capture audio using `chrome.tabCapture.capture()` API
4. Convert audio to 16kHz mono WAV format
5. Send chunks every ~500ms to `/stream/push`
6. Poll `/stream/poll` every 200-500ms for results
7. Display `segments` (finalized) and `current` (partial) in UI
8. Handle 404 errors (session expired) by recreating session
9. DELETE session when stopping capture

### Audio Format Requirements

- Format: WAV
- Sample rate: 16000 Hz
- Channels: Mono (1 channel)
- Bit depth: 16-bit or 32-bit float

### Example JavaScript (Web Audio API)

```javascript
// Convert AudioBuffer to 16kHz mono WAV
function audioBufferToWav(buffer, targetSampleRate = 16000) {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;

  // Resample to 16kHz
  const ratio = sampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);

  // Simple linear interpolation resampling
  for (let i = 0; i < newLength; i++) {
    const srcIdx = i * ratio;
    const idx = Math.floor(srcIdx);
    const frac = srcIdx - idx;
    const sample = buffer.getChannelData(0);
    result[i] = sample[idx] * (1 - frac) + (sample[idx + 1] || 0) * frac;
  }

  // Create WAV file
  const wavBuffer = new ArrayBuffer(44 + result.length * 2);
  const view = new DataView(wavBuffer);

  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + result.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, result.length * 2, true);

  // Write samples
  const offset = 44;
  for (let i = 0; i < result.length; i++) {
    const s = Math.max(-1, Math.min(1, result[i]));
    view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}
```

## Session Lifecycle

1. **Create**: Client creates session with `/stream/create`
2. **Stream**: Client pushes audio chunks via `/stream/push`
3. **Poll**: Client polls `/stream/poll` for transcription results
4. **Close**: Session is closed via DELETE or auto-expires after timeout

Sessions automatically expire after 60 seconds of inactivity (configurable via `--timeout`).
Push and poll requests to expired sessions return 404 errors.

## How VAD Works

The server uses energy-based Voice Activity Detection:

1. Audio is high-pass filtered (100 Hz cutoff) to remove low-frequency noise
2. RMS energy is calculated for each audio chunk
3. If energy exceeds threshold (0.01 default), speech is detected
4. Transcription runs only when speech is detected
5. After a period of silence, the segment is finalized

This prevents the model from hallucinating text during silence (e.g., "Thanks for watching").
