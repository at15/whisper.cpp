# Stream Server using Websocket

## Background

I want to want do live caption on audio stream captured from browser tab.
However, existing http and stream examples do not work out of box:

- http server does not support stream, browser side need to chunk audio, e.g. every 2 seconds
- stream example does not have http server and requires getting audio from the microphone.

Without stream and VAD, the transcript generate a lot of garbage
such as `Thanks for watching` and `Please like and subscribe`.

The webassembly example does not work because we want to run the server
with GPU support.

There are also python examples implemeting vad such as 

- `/Users/at15/w/src/github.com/fann1993814/whisper.cpy/whispercpy/stream.py`

I want to implement stream and VAD using http server.
We can simulate websocket by making the client stateful with polling.
For example, when sending the request, client side should create and provide ids for:

- a session id to identify the audio stream, this allow different clients to use same server
- a chunk id to identify the order of chunk

We split the audio upload and text retrieval into two APIs.
The audio upload API just return without waiting on the generation.
The text retrieval API poll the text generation result.
The text retrieval API should provide proper cursor e.g. timestamp
so client side can fetch the proper data.

NOTE: You need to think harder on the stream and VAD logic, mine is just for reference and may not be accurate.

## Instructions for claude code

- Read related code and write down your plan in a markdown file `stream.claude.md`
- Update the build logic so we can build our server like other examples
- Provide proper doc so the chrome extension knows how to use this server
- Provide enough log for debugging, we need to test it manually
