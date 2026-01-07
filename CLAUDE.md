# ElevenLabs Agent Configuration

## Project Overview

This is an **experimental project** for creating a custom LLM proxy for ElevenLabs agents.

**Dependencies:** This project uses UV for dependency management. Dependencies are defined **inline** in the `openai_proxy.py` file using the PEP 723 inline script metadata format (see the `# /// script` block at the top of the file).

## Development Guidelines

**IMPORTANT:** Every time you update `openai_proxy_xxx.py`, you MUST check the server logs to verify that:
1. The server reloaded successfully without errors
2. No import errors or dependency issues occurred
3. The changes are working as expected

The server runs with `--reload` flag, so changes should trigger an automatic reload. Check the logs for:
- `Watching for file changes` (indicates reload is active)
- Any Python tracebacks or error messages
- Successful startup messages

If the server is running with log capture, read the `server.log` file to verify the reload.

**Code Quality:** After each change, run `uv tool run ruff format <updated-file>`, `uv tool run ruff check --fix <updated-file>`, and `uv tool run pyright <updated-file>` to ensure the code is formatted and linted correctly.

### Research and Verification

**CRITICAL:** ALWAYS use Perplexity search when you don't have a high level of certainty about how to implement something. This is MANDATORY, not optional.

You MUST use Perplexity for:
- API endpoints, request/response formats, and parameter names
- Library-specific syntax and features (e.g., Loguru colors, LiteLLM parameters)
- Model capabilities and available features
- Best practices for libraries you're using
- Any technical detail where you're not 100% certain

**DO NOT GUESS. DO NOT ASSUME. ALWAYS VERIFY WITH PERPLEXITY FIRST.**

## Testing APIs - HTTPie Preference

**Always use HTTPie instead of curl for testing APIs.**

HTTPie provides better readability, automatic JSON formatting, and a more intuitive syntax.

### Installation
```bash
# macOS
brew install httpie

# Ubuntu/Debian
apt install httpie

# pip
pip install httpie
```

### HTTPie Interactive Mode

Use `--print=HhBb` for full interactive mode showing:
- `H` - Request headers
- `h` - Response headers
- `B` - Request body
- `b` - Response body

**Example:**
```bash
# Testing the OpenAI proxy
http --print=HhBb POST https://yorrick.ngrok.dev/v1/chat/completions \
  Content-Type:application/json \
  messages:='[{"role":"user","content":"test"}]' \
  model=gpt-4 \
  stream:=true

# Testing ElevenLabs API
http --print=HhBb GET https://api.elevenlabs.io/v1/convai/agents/agent_2501k520e4p6eqs87ga1ke8a67fs \
  xi-api-key:YOUR_API_KEY
```

### HTTPie Tips
- Headers: `Header-Name:value` (note the `:` without space)
- JSON fields: `field=value` or `field:=json_value`
- Boolean/Number: Use `:=` for raw JSON (e.g., `stream:=true`)
- No need for `-X GET/POST` - HTTPie infers from data presence

## Fetching Agent Configuration

### Using the ElevenLabs API

To fetch your agent's configuration definition, use the ElevenLabs API:

**Endpoint:**
```
GET https://api.elevenlabs.io/v1/convai/agents/{agent_id}
```

**Example using HTTPie (PREFERRED):**
```bash
# Standard mode
http GET https://api.elevenlabs.io/v1/convai/agents/agent_2501k520e4p6eqs87ga1ke8a67fs \
  xi-api-key:YOUR_ELEVENLABS_API_KEY

# Interactive mode (recommended for exploring APIs)
http --print=HhBb GET https://api.elevenlabs.io/v1/convai/agents/agent_2501k520e4p6eqs87ga1ke8a67fs \
  xi-api-key:YOUR_ELEVENLABS_API_KEY
```

**Example using curl:**
```bash
curl -X GET "https://api.elevenlabs.io/v1/convai/agents/agent_2501k520e4p6eqs87ga1ke8a67fs" \
  -H "xi-api-key: YOUR_ELEVENLABS_API_KEY"
```

**Example using Python:**
```python
import requests

agent_id = "agent_2501k520e4p6eqs87ga1ke8a67fs"
api_key = "YOUR_ELEVENLABS_API_KEY"

response = requests.get(
    f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}",
    headers={"xi-api-key": api_key}
)

agent_config = response.json()
print(json.dumps(agent_config, indent=2))
```

### Finding Your Agent ID

Your agent ID can be found in:
1. The ElevenLabs dashboard URL when viewing the agent
2. The agent configuration JSON under the `agent_id` field
3. Example: `agent_2501k520e4p6eqs87ga1ke8a67fs`

### Key Configuration Fields for Custom LLM

When setting up a custom LLM, the important fields are:

```json
{
  "conversation_config": {
    "agent": {
      "prompt": {
        "custom_llm": {
          "url": "https://your-server.ngrok.dev/v1",
          "model_id": "gpt-4",
          "api_key": {
            "secret_id": "your_secret_id"
          },
          "api_type": "chat_completions"
        }
      }
    }
  }
}
```

**Important Notes:**
- `url`: Should be your base URL + `/v1` (ElevenLabs appends `/chat/completions`)
- `model_id`: Must be a valid OpenAI model name (e.g., `gpt-4`, `gpt-3.5-turbo`, `gpt-4o`)
- `api_key.secret_id`: Reference to a secret stored in ElevenLabs
- `api_type`: Should be `"chat_completions"` for OpenAI-compatible APIs

### API Documentation

Full ElevenLabs API documentation: https://elevenlabs.io/docs/api-reference/agents

## Running Agent Tests

You can run tests against your agent to verify its behavior.

**Endpoint:**

```http
POST https://api.elevenlabs.io/v1/convai/agents/{agent_id}/run-tests
```

**Example using HTTPie with environment variables:**

```bash
echo '{"tests": [{"test_id": "'$ELEVENLABS_TEST_ID'"}]}' | \
  http POST https://api.elevenlabs.io/v1/convai/agents/$ELEVENLABS_AGENT_ID/run-tests \
  "xi-api-key: $ELEVENLABS_API_KEY"
```

**Example using HTTPie with inline values:**

```bash
http POST https://api.elevenlabs.io/v1/convai/agents/agent_2501k520e4p6eqs87ga1ke8a67fs/run-tests \
  xi-api-key:YOUR_ELEVENLABS_API_KEY \
  tests:='[{"test_id": "your_test_id"}]'
```

**Required environment variables:**

- `ELEVENLABS_AGENT_ID`: Your agent ID (e.g., `agent_2501k520e4p6eqs87ga1ke8a67fs`)
- `ELEVENLABS_TEST_ID`: The test ID you want to run
- `ELEVENLABS_API_KEY`: Your ElevenLabs API key

## Troubleshooting Custom LLM

### No requests reaching your server?

1. **Check ngrok web interface** at http://127.0.0.1:4040
   - Shows ALL requests hitting ngrok, even failed ones
   - If nothing appears here, ElevenLabs isn't sending requests

2. **Verify configuration is saved**
   - Re-fetch the agent config via API to confirm your changes were saved
   - Check that `"llm": "custom-llm"` is set

3. **Verify API key secret exists**
   - The `secret_id` must reference a valid secret in ElevenLabs
   - Empty/invalid secrets may prevent requests

4. **Test with webhook.site**
   - Use https://webhook.site as a temporary URL
   - See if ElevenLabs sends any requests there
   - Helps isolate if issue is with ElevenLabs or your server

### Current Setup

- **ngrok URL:** https://yorrick.ngrok.dev
- **Local server:** http://localhost:8013
- **Custom LLM URL:** https://yorrick.ngrok.dev/v1
- **Full endpoint:** https://yorrick.ngrok.dev/v1/chat/completions
- **Agent ID:** agent_2501k520e4p6eqs87ga1ke8a67fs
