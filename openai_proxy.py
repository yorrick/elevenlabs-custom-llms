# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "fastapi",
#   "openai",
#   "uvicorn",
#   "python-dotenv",
#   "loguru",
# ]
# ///

from importlib import reload
import json
import os
import fastapi
from fastapi import Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = fastapi.FastAPI()
oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    logger.info(f"Headers: {dict(request.headers)}")

    # Read and log body for POST requests
    if request.method == "POST":
        body = await request.body()
        logger.info(f"Body: {body.decode()}")

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
    logger.info(f"Request: model={request.model}, stream={request.stream}, messages={len(request.messages)}")

    oai_request = request.dict(exclude_none=True)
    if "user_id" in oai_request:
        oai_request["user"] = oai_request.pop("user_id")

    chat_completion_coroutine = await oai_client.chat.completions.create(**oai_request)

    async def event_stream():
        try:
            async for chunk in chat_completion_coroutine:
                # Convert the ChatCompletionChunk to a dictionary before JSON serialization
                chunk_dict = chunk.model_dump()
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("openai_proxy:app", host="0.0.0.0", port=8013, reload=True)
