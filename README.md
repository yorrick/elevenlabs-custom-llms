# OpenAI Proxy Server

A FastAPI-based proxy server for OpenAI chat completions API with support for streaming responses.

## Prerequisites

- [UV](https://docs.astral.sh/uv/) installed
- OpenAI API key

## Setup

1. Create a `.env` file in the project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Server

The script uses UV's inline dependencies feature, so you can run it with a single command:

```bash
uv run openai_proxy.py
```

UV will automatically install all required dependencies (fastapi, openai, uvicorn, python-dotenv) in an isolated environment and run the server.

The server will start on `http://0.0.0.0:8013`

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

## Features

- OpenAI chat completions proxy
- Streaming response support
- Optional user_id parameter (mapped to OpenAI's user parameter)
- Environment-based API key configuration
