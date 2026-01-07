# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "fastapi",
#   "litellm",
#   "uvicorn",
#   "python-dotenv",
#   "loguru",
# ]
# ///

import json
import os
import sys
import fastapi
from fastapi import Request
from fastapi.responses import StreamingResponse
import litellm
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Configure logger: Use LOGURU_LEVEL env var, default to INFO
logger.remove()  # Remove default handler
log_level = os.getenv("LOGURU_LEVEL", "INFO")
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)

# Retrieve API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Set LiteLLM API key for Gemini
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

app = fastapi.FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    logger.debug(f"Headers: {dict(request.headers)}")

    # Read and log body for POST requests
    if request.method == "POST":
        body = await request.body()
        logger.debug(f"Body: {body.decode()}")

        # Log conversation messages at INFO level with colors
        try:
            body_json = json.loads(body.decode())
            if "messages" in body_json:
                for msg in body_json["messages"]:
                    role = msg.get("role", "unknown")
                    if role == "user":
                        content = msg.get("content", "")
                        logger.opt(colors=True).info(f"<cyan>👤 USER: {content}</cyan>")
                    elif role == "assistant":
                        content = msg.get("content", "")
                        logger.opt(colors=True).info(
                            f"<green>🤖 ASSISTANT: {content}</green>"
                        )
        except Exception as e:
            logger.debug(f"Could not parse messages: {e}")

        # Create a new request with the body since we consumed it
        from starlette.requests import Request as StarletteRequest

        async def receive():
            return {"type": "http.request", "body": body}

        request = StarletteRequest(request.scope, receive)

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = "medium"
    user_id: Optional[str] = None


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> StreamingResponse:
    logger.info(
        f"Request: model={request.model}, stream={request.stream}, "
        f"reasoning_effort={request.reasoning_effort}, messages={len(request.messages)}"
    )

    # Convert messages to dict format
    messages = [msg.dict() for msg in request.messages]

    # Pass request to LiteLLM
    litellm_request = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
        "reasoning_effort": request.reasoning_effort,
        "drop_params": True,  # Silently drop unsupported params like reasoning_effort
    }

    if request.temperature is not None:
        litellm_request["temperature"] = request.temperature
    if request.max_tokens is not None:
        litellm_request["max_tokens"] = request.max_tokens

    response = await litellm.acompletion(**litellm_request)

    async def event_stream():
        try:
            async for chunk in response:
                # LiteLLM returns OpenAI-compatible chunks
                # Convert to dict if needed
                if hasattr(chunk, "model_dump"):
                    chunk_dict = chunk.model_dump()
                elif hasattr(chunk, "dict"):
                    chunk_dict = chunk.dict()
                else:
                    chunk_dict = dict(chunk)

                # Log delta keys and content
                if "choices" in chunk_dict:
                    for choice in chunk_dict["choices"]:
                        if "delta" in choice:
                            delta = choice["delta"]
                            # Log reasoning/thinking content
                            if (
                                "reasoning_content" in delta
                                and delta["reasoning_content"]
                            ):
                                logger.opt(colors=True).info(
                                    f"<magenta>🧠 REASONING: {delta['reasoning_content']}</magenta>"
                                )
                            # Log actual content
                            if "content" in delta and delta["content"]:
                                logger.opt(colors=True).info(
                                    f"<green>🤖 ASSISTANT: {delta['content']}</green>"
                                )

                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import os

    # Get module name from file name (without .py extension)
    # This allows the file to be renamed without updating the uvicorn.run() call
    # We need to pass an import string (not the app object) to enable reload
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8013, reload=True)
