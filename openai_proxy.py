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

from importlib import reload
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

# Configure logger: INFO level by default, DEBUG for detailed request bodies
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")

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

        # Log conversation messages at INFO level
        try:
            body_json = json.loads(body.decode())
            if "messages" in body_json:
                for msg in body_json["messages"]:
                    role = msg.get("role", "unknown")
                    if role in ["user", "assistant"]:
                        content = msg.get("content", "")
                        logger.info(f"Conversation [{role}]: {content}")
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


@app.get("/")
async def health_check():
    logger.info("Health check endpoint hit")
    return {"status": "ok", "service": "openai-proxy"}


@app.get("/v1")
async def health_check_v1():
    logger.info("V1 health check endpoint hit")
    return {"status": "ok", "service": "openai-proxy", "version": "v1"}


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user_id: Optional[str] = None


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> StreamingResponse:
    logger.info(
        f"Request: model={request.model}, stream={request.stream}, messages={len(request.messages)}"
    )

    # Convert messages to dict format
    messages = [msg.dict() for msg in request.messages]

    # Use Gemini Flash via LiteLLM
    litellm_request = {
        "model": "gemini/gemini-2.5-flash",  # Gemini 3.0 Flash
        "messages": messages,
        "stream": request.stream,
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
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("openai_proxy:app", host="0.0.0.0", port=8013, reload=True)
