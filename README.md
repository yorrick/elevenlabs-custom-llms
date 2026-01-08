# OpenAI Proxy Server

Back your ElevenLabs agents with code, using an OpenAI compatible API.

## Prerequisites

- [UV](https://docs.astral.sh/uv/) installed
- OpenAI/Gemini/... API key

## Setup

1. Create a `.env` file in the project directory:
```env
GEMINI_API_KEY=your_api_key_here

# Optional: For LangFuse observability (only needed for openai_proxy_with_langfuse.py)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_OTEL_HOST=https://cloud.langfuse.com
```

### LangFuse Setup (Optional)

To use the LangFuse integration (`openai_proxy_with_langfuse.py`), you need to:

1. Sign up for a free account at [LangFuse](https://cloud.langfuse.com)
2. Create a new project in the LangFuse dashboard
3. Go to Project Settings to get your API keys
4. Add the keys to your `.env` file (see above)

**What LangFuse provides:**
- Full request/response tracing
- Token usage tracking
- Latency metrics (first token, full response)
- Streaming chunk timing
- Error monitoring
- User-level analytics (when `user_id` is provided)

The proxy will work without LangFuse configured, but tracing will be disabled.

## Available Proxy Examples

This repository includes multiple proxy examples:

1. **`openai_proxy_simple.py`** - Basic OpenAI-compatible proxy
2. **`openai_proxy_with_langfuse.py`** - Proxy with LangFuse observability integration
3. **`openai_proxy_with_cot.py`** - Proxy with chain-of-thought reasoning
4. **`openai_proxy_with_notifications.py`** - Proxy with notifications support

## Running the Server

The scripts use UV's inline dependencies feature, so you can run any of them with a single command:

```bash
# Simple proxy
uv run openai_proxy_simple.py

# With LangFuse tracing
uv run openai_proxy_with_langfuse.py

# With chain-of-thought
uv run openai_proxy_with_cot.py

# With notifications
uv run openai_proxy_with_notifications.py
```

UV will automatically install all required dependencies in an isolated environment and run the server.

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
- LangFuse observability and tracing (optional)
