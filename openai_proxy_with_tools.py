# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "fastapi",
#   "litellm>=1.80.5",
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
from fastapi.responses import StreamingResponse, JSONResponse
import litellm
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

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

# Static inventory data for the three store locations
INVENTORY_DATA = {
    "f47ac10b-58cc-4372-a567-0e02b2c3d479": {  # Oakville
        "store_name": "Oakville",
        "inventory": {
            "dewalt 20v max cordless drill": {
                "in_stock": True,
                "quantity": 24,
                "price": 179.99,
                "aisle": "B12",
            },
            "pressure-treated 2x4": {
                "in_stock": True,
                "quantity": 450,
                "price": 8.99,
                "unit": "each",
                "aisle": "Lumber Yard",
            },
            "milwaukee m18 impact driver": {
                "in_stock": True,
                "quantity": 15,
                "price": 199.99,
                "aisle": "B14",
            },
            "ryobi circular saw": {
                "in_stock": False,
                "quantity": 0,
                "price": 89.99,
                "aisle": "B10",
            },
            "gorilla glue wood glue": {
                "in_stock": True,
                "quantity": 87,
                "price": 6.49,
                "aisle": "A5",
            },
            "3m n95 respirator masks": {
                "in_stock": True,
                "quantity": 120,
                "price": 24.99,
                "unit": "box of 10",
                "aisle": "C3",
            },
            "exterior latex paint": {
                "in_stock": True,
                "quantity": 68,
                "price": 42.99,
                "unit": "gallon",
                "aisle": "Paint Department",
            },
            "husky tool box": {
                "in_stock": True,
                "quantity": 12,
                "price": 149.99,
                "aisle": "D8",
            },
        },
    },
    "d8e8f8f8-3d3d-4c4c-8c8c-8c8c8c8c8c8c": {  # Burnaby
        "store_name": "Burnaby",
        "inventory": {
            "dewalt 20v max cordless drill": {
                "in_stock": True,
                "quantity": 18,
                "price": 179.99,
                "aisle": "Power Tools Section",
            },
            "pressure-treated 2x4": {
                "in_stock": True,
                "quantity": 320,
                "price": 8.99,
                "unit": "each",
                "aisle": "Lumber Yard",
            },
            "milwaukee m18 impact driver": {
                "in_stock": False,
                "quantity": 0,
                "price": 199.99,
                "aisle": "Power Tools Section",
            },
            "ryobi circular saw": {
                "in_stock": True,
                "quantity": 9,
                "price": 89.99,
                "aisle": "Power Tools Section",
            },
            "gorilla glue wood glue": {
                "in_stock": True,
                "quantity": 45,
                "price": 6.49,
                "aisle": "Adhesives",
            },
            "3m n95 respirator masks": {
                "in_stock": True,
                "quantity": 200,
                "price": 24.99,
                "unit": "box of 10",
                "aisle": "Safety Equipment",
            },
            "exterior latex paint": {
                "in_stock": True,
                "quantity": 92,
                "price": 42.99,
                "unit": "gallon",
                "aisle": "Paint Department",
            },
            "makita cordless combo kit": {
                "in_stock": True,
                "quantity": 8,
                "price": 349.99,
                "aisle": "Power Tools Section",
            },
        },
    },
    "123e4567-e89b-12d3-a456-426614174000": {  # Halifax
        "store_name": "Halifax",
        "inventory": {
            "dewalt 20v max cordless drill": {
                "in_stock": True,
                "quantity": 31,
                "price": 179.99,
                "aisle": "B8",
            },
            "pressure-treated 2x4": {
                "in_stock": True,
                "quantity": 580,
                "price": 8.99,
                "unit": "each",
                "aisle": "Lumber Yard",
            },
            "milwaukee m18 impact driver": {
                "in_stock": True,
                "quantity": 22,
                "price": 199.99,
                "aisle": "B8",
            },
            "ryobi circular saw": {
                "in_stock": True,
                "quantity": 14,
                "price": 89.99,
                "aisle": "B12",
            },
            "gorilla glue wood glue": {
                "in_stock": True,
                "quantity": 103,
                "price": 6.49,
                "aisle": "A3",
            },
            "3m n95 respirator masks": {
                "in_stock": False,
                "quantity": 0,
                "price": 24.99,
                "unit": "box of 10",
                "aisle": "C2",
            },
            "exterior latex paint": {
                "in_stock": True,
                "quantity": 134,
                "price": 42.99,
                "unit": "gallon",
                "aisle": "Paint Center",
            },
            "craftsman lawn mower": {
                "in_stock": True,
                "quantity": 6,
                "price": 399.99,
                "aisle": "Garden Center",
            },
        },
    },
}


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
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    user_id: Optional[str] = None


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> StreamingResponse:
    logger.info(
        f"Request: model={request.model}, stream={request.stream}, "
        f"messages={len(request.messages)}, tools={len(request.tools) if request.tools else 0}"
    )

    # Convert messages to dict format
    messages = [msg.dict(exclude_none=True) for msg in request.messages]

    # Pass request to LiteLLM - simple pass-through including tools
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
    if request.tools is not None:
        litellm_request["tools"] = request.tools
    if request.tool_choice is not None:
        litellm_request["tool_choice"] = request.tool_choice

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
                            if "tool_calls" in delta and delta["tool_calls"]:
                                logger.opt(colors=True).info(
                                    f"<yellow>🔧 TOOL CALL: {delta['tool_calls']}</yellow>"
                                )

                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def fuzzy_match_product(query: str, product_list: List[str]) -> Optional[str]:
    """
    Use LLM to fuzzy match user query to available products.
    Returns the best matching product name, or None if no good match.
    """
    products_text = "\n".join([f"- {p}" for p in product_list])

    prompt = f"""You are a product matching assistant for a hardware store.

User is searching for: "{query}"

Available products:
{products_text}

Task: Return ONLY the exact product name from the list above that best matches the user's search query. If no product is a reasonable match, return "NO_MATCH".

Examples:
- "two by fours" -> "pressure-treated 2x4"
- "dewalt drill" -> "dewalt 20v max cordless drill"
- "paint" -> "exterior latex paint"

Return ONLY the product name or "NO_MATCH", nothing else."""

    try:
        response = await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0.1,
        )

        match = response.choices[0].message.content.strip()

        if match == "NO_MATCH" or match not in product_list:
            logger.info(f"LLM fuzzy match: '{query}' -> NO_MATCH")
            return None

        logger.opt(colors=True).info(
            f"<cyan>🔍 LLM fuzzy match: '{query}' -> '{match}'</cyan>"
        )
        return match

    except Exception as e:
        logger.error(f"LLM fuzzy matching failed: {e}")
        return None


@app.get("/tools/inventory_check")
async def inventory_check(item_name: str, store_id: str) -> JSONResponse:
    """
    Tool endpoint for checking inventory at a specific store location.
    This endpoint is called by ElevenLabs when the agent decides to check inventory.
    ElevenLabs calls this with GET and query parameters.
    """
    logger.opt(colors=True).info(
        f"<yellow>🔧 Inventory check: item='{item_name}', store_id='{store_id}'</yellow>"
    )

    # Check if store exists
    if store_id not in INVENTORY_DATA:
        error_response = {
            "error": "Store not found",
            "message": f"No inventory data available for store ID: {store_id}",
        }
        logger.warning(f"Store not found: {store_id}")
        return JSONResponse(content=error_response, status_code=404)

    store_data = INVENTORY_DATA[store_id]
    store_name = store_data["store_name"]

    # Get list of all products in this store
    product_list = list(store_data["inventory"].keys())

    # Use LLM to fuzzy match the query to available products
    matched_product = await fuzzy_match_product(item_name, product_list)

    if not matched_product:
        # Item not found
        response = {
            "store_name": store_name,
            "store_id": store_id,
            "item_searched": item_name,
            "found": False,
            "message": f"Sorry, we don't have '{item_name}' in our system for the {store_name} location.",
        }
        logger.info(f"Item not found: {item_name}")
        return JSONResponse(content=response)

    # Item found - return the matched product
    product_data = store_data["inventory"][matched_product]

    response = {
        "store_name": store_name,
        "store_id": store_id,
        "item_name": matched_product,
        "in_stock": product_data["in_stock"],
        "quantity": product_data["quantity"],
        "price": product_data["price"],
        "aisle": product_data["aisle"],
    }

    # Add unit if present
    if "unit" in product_data:
        response["unit"] = product_data["unit"]

    logger.opt(colors=True).info(
        f"<green>✓ Found: {matched_product} - In stock: {product_data['in_stock']} (Qty: {product_data['quantity']})</green>"
    )

    return JSONResponse(content=response)


if __name__ == "__main__":
    import os

    # Get module name from file name (without .py extension)
    # This allows the file to be renamed without updating the uvicorn.run() call
    # We need to pass an import string (not the app object) to enable reload
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8013, reload=True)
