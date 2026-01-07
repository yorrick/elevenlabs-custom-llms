# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "fastapi",
#   "litellm",
#   "uvicorn",
#   "python-dotenv",
#   "loguru",
#   "aiosqlite",
#   "twilio",
#   "python-multipart",
# ]
# ///

import json
import os
import sys
import warnings
import fastapi
from fastapi import Request, Form
from fastapi.responses import StreamingResponse, Response
import litellm
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import aiosqlite
from datetime import datetime
from contextlib import asynccontextmanager

# Suppress Pydantic serialization warnings from LiteLLM streaming
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")

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

# Database file path
DB_PATH = os.getenv("DB_PATH", "notifications.db")


async def init_db():
    """Initialize the SQLite database with notifications table"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_number TEXT NOT NULL,
                to_number TEXT NOT NULL,
                message TEXT NOT NULL,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_read INTEGER DEFAULT 0
            )
        """)
        await db.commit()
    logger.info("Database initialized")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    await init_db()
    yield
    # Shutdown (cleanup if needed)


app = fastapi.FastAPI(lifespan=lifespan)


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


@app.post("/webhook/twilio/sms")
async def twilio_sms_webhook(
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(...),
):
    """
    Webhook endpoint for Twilio SMS.
    Twilio sends form-encoded data with From, To, Body fields.
    """
    logger.opt(colors=True).info(
        f"<yellow>📱 SMS received from {From} to {To}: {Body}</yellow>"
    )

    # Store notification in database
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO notifications (from_number, to_number, message, received_at)
            VALUES (?, ?, ?, ?)
            """,
            (From, To, Body, datetime.utcnow().isoformat()),
        )
        await db.commit()

    logger.info("SMS notification stored in database")

    # Respond to Twilio with empty TwiML (no response message)
    return Response(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )


async def get_unread_notifications():
    """Retrieve all unread notifications from the database"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, from_number, to_number, message, received_at
            FROM notifications
            WHERE is_read = 0
            ORDER BY received_at ASC
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def mark_notifications_as_read(notification_ids: List[int]):
    """Mark notifications as read"""
    if not notification_ids:
        return

    async with aiosqlite.connect(DB_PATH) as db:
        placeholders = ",".join("?" * len(notification_ids))
        await db.execute(
            f"UPDATE notifications SET is_read = 1 WHERE id IN ({placeholders})",
            notification_ids,
        )
        await db.commit()


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
        f"Request: model={request.model}, stream={request.stream}, "
        f"messages={len(request.messages)}"
    )

    # Get unread notifications
    unread_notifications = await get_unread_notifications()

    # Convert messages to dict format
    messages = [msg.dict() for msg in request.messages]

    # Inject unread notifications into the conversation
    if unread_notifications:
        notification_ids = [notif["id"] for notif in unread_notifications]

        # Inject SMS messages as user messages (from the client)
        # Insert them at the end of the conversation, just before the latest user message
        for notif in unread_notifications:
            user_message = {
                "role": "user",
                "content": f"[SMS from client]: {notif['message']}",
            }
            # Insert before the last user message
            messages.insert(-1 if messages else 0, user_message)

        logger.opt(colors=True).info(
            f"<yellow>📬 Injecting {len(unread_notifications)} SMS from client</yellow>"
        )

        # Mark notifications as read
        await mark_notifications_as_read(notification_ids)

    # Pass request to LiteLLM - simple pass-through
    litellm_request = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
        "drop_params": True,  # Silently drop unsupported params
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

                # Log content only
                if "choices" in chunk_dict:
                    for choice in chunk_dict["choices"]:
                        if "delta" in choice:
                            delta = choice["delta"]
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
