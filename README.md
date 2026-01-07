# OpenAI Proxy Server

Back your ElevenLabs agents with code, using an OpenAI compatible API.

## Prerequisites

- [UV](https://docs.astral.sh/uv/) installed
- OpenAI/Gemini/... API key

## Setup

1. Create a `.env` file in the project directory:
```env
GEMINI_API_KEY=your_api_key_here
```

## Running the Server

The script uses UV's inline dependencies feature, so you can run it with a single command:

```bash
uv run openai_proxy_simple.py
```

UV will automatically install all required dependencies (fastapi, litellm, uvicorn, python-dotenv, loguru) in an isolated environment and run the server.

The server will start on `http://127.0.0.1:8013`

### Logging

To capture server logs to both console and file (for debugging or for Claude Code to read):

```bash
# Redirect all output to both console and file
uv run openai_proxy_simple.py 2>&1 | tee server.log
```

This uses `tee` to write logs to both:
- **Console (stdout)**: So you can see them in real-time
- **File (`server.log`)**: So Claude Code can read and analyze them

The `2>&1` redirects stderr to stdout so all log messages are captured.

## API Usage

### Endpoint

```
POST /v1/chat/completions
```

### Request Body

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true,
  "user_id": "optional_user_id"
}
```

### Example using cURL

```bash
curl -X POST http://localhost:8013/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt-4",
    "stream": true
  }'
```

## Testing with ElevenLabs

To trigger ElevenLabs agent tests, use the following HTTPie command:

```bash
echo '{"tests": [{"test_id": "'$ELEVENLABS_TEST_ID'"}]}' | \
  http POST https://api.elevenlabs.io/v1/convai/agents/$ELEVENLABS_AGENT_ID/run-tests \
  "xi-api-key: $ELEVENLABS_API_KEY"
```

Required environment variables:
- `ELEVENLABS_AGENT_ID`: Your agent ID (e.g., `agent_2501k520e4p6eqs87ga1ke8a67fs`)
- `ELEVENLABS_TEST_ID`: The test ID you want to run
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key

## Features

- OpenAI chat completions proxy
- Streaming response support
- Optional user_id parameter (mapped to OpenAI's user parameter)
- Environment-based API key configuration
