import os
import time
from typing import List, Optional, Union, Dict, Any

import json

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import logging

load_dotenv()

# Configuration from environment variables
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
CLIENT_ID = os.environ.get("CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", "")
PORT = int(os.environ.get("PORT", "8081"))
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "60"))

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

app = FastAPI(title="OpenAI-compatible shim for AcademicAI API", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: Union[str, Dict[str, Any], List[Any]]


class ResponseFormat(BaseModel):
    # Only "json_object" or "text" allowed; validated in build_upstream_payload
    type: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


# ==================================================
# ========(old) Chat Completion API endpoint========
# ==================================================


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    stream: Optional[bool] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


def config_ok() -> Optional[JSONResponse]:
    """Return a JSONResponse with code 500 if config is invalid; None if okay"""

    missing = []
    if not BASE_URL:
        missing.append("BASE_URL")
    if not CLIENT_ID:
        missing.append("CLIENT_ID")
    if not CLIENT_SECRET:
        missing.append("CLIENT_SECRET")
    if missing:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Missing required environment variables: {', '.join(missing)}",
                    "type": "config_error",
                    "code": 500,
                }
            },
        )
    return None


def build_upstream_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    """Validate and build the payload to send to the upstream AcademicAI API"""

    # Validate response_format.type if provided
    if req.response_format and req.response_format.type not in (
        None,
        "json_object",
        "text",
    ):
        raise ValueError('response_format.type must be "json_object" or "text"')

    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["maxTokens"] = req.max_tokens
    if req.stop is not None:
        payload["stop"] = req.stop
    if req.frequency_penalty is not None:
        payload["frequencyPenalty"] = req.frequency_penalty
    if req.presence_penalty is not None:
        payload["presencePenalty"] = req.presence_penalty
    if req.seed is not None:
        payload["seed"] = req.seed
    if req.response_format is not None and req.response_format.type is not None:
        payload["responseFormat"] = {"type": req.response_format.type}

    if req.tools is not None:
        payload["tools"] = [t.model_dump() for t in req.tools]
    if req.tool_choice is not None:
        payload["toolChoice"] = req.tool_choice  # or "tool_choice" depending on upstream API casing

    return payload


def map_upstream_to_openai_like(
    upstream_json: Dict[str, Any],
    request_model: str,
) -> Dict[str, Any]:
    # Expected upstream:
    # {
    #   "data": {
    #       "model": "...",
    #       "role": "assistant",
    #       "content": "...",
    #       "finishReason": "stop",
    #       "usage": {"promptTokens": n, "completionTokens": n, "totalTokens": n}
    #   }
    # }
    # TODO: If upstream returns a different shape (e.g., multiple messages/choices),
    #       adjust parsing; guarded by .get defaults to avoid crashes.

    data = upstream_json.get("data") or {}
    role = data.get("role", "assistant")
    content = data.get("content", "")
    finish_reason = data.get("finishReason", "stop")
    model = data.get("model") or request_model
    usage = data.get("usage") or {}

    prompt_tokens = usage.get("promptTokens", 0)
    completion_tokens = usage.get("completionTokens", 0)
    total_tokens = usage.get("totalTokens", prompt_tokens + completion_tokens)

    created_ts = int(time.time())
    created_ms = int(time.time() * 1000)

    # ---- NEW: parse tool_calls from upstream ----
    # Adjust the keys below to match what YOUR upstream API actually returns.
    # Common shapes: data.toolCalls, data.tool_calls, data.functionCall
    raw_tool_calls = data.get("toolCalls") or data.get("tool_calls")

    message: Dict[str, Any] = {
        "role": role,
        "content": content,  # can be None when tool_calls are present
    }

    if raw_tool_calls:
        # Normalize to OpenAI tool_calls format
        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            # Handle both camelCase and snake_case from upstream
            func = tc.get("function") or {}
            tool_calls.append({
                "id": tc.get("id") or f"call_{created_ms}_{i}",
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),  # must be a JSON string
                },
            })
        message["tool_calls"] = tool_calls
        # When tool_calls are present, content may be null
        if not content:
            message["content"] = None
        # OpenAI uses "tool_calls" as the finish_reason when tools are invoked
        finish_reason = "tool_calls"


    openai_like = {
        "id": f"chatcmpl-{created_ms}",
        "object": "chat.completion",
        "created": created_ts,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    return openai_like


# Default (old) OpenAI-compatible endpoint for chat completions
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):

    # Force streaming to False for now, since we don't support it
    if req.stream is True:
        logger.debug(
            "Streaming not supported in /v1/chat/completions; forcing non-streaming for now"
        )
        req.stream = False

    cfg_err = config_ok()
    if cfg_err is not None:
        return cfg_err

    try:
        payload = build_upstream_payload(req)
    except ValueError as e:
        logger.debug(f"Invalid request: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    upstream_url = f"{BASE_URL}/api/v1/llm/chat"
    headers = {
        "X-Client-ID": CLIENT_ID,
        "X-Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(upstream_url, headers=headers, json=payload)

            # If upstream NOT 2xx, synthesize OpenAI-like error
            if resp.status_code < 200 or resp.status_code >= 300:
                # Try to extract JSON --> fallback is text
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                return JSONResponse(
                    status_code=resp.status_code,
                    content={
                        "error": {
                            "message": detail,
                            "type": "upstream_error",
                            "code": resp.status_code,
                        }
                    },
                )

            upstream_json = resp.json()
            mapped = map_upstream_to_openai_like(upstream_json, request_model=req.model)
            return JSONResponse(status_code=200, content=mapped)

        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": f"Upstream timeout after {TIMEOUT_SECONDS} seconds",
                        "type": "upstream_error",
                        "code": 504,
                    }
                },
            )

        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"HTTP error contacting upstream: {str(e)}",
                        "type": "upstream_error",
                        "code": 502,
                    }
                },
            )


# ===========================================
# ========(newer) Response API endpoint======
# ===========================================


class ResponsesInputMessage(BaseModel):
    role: str
    content: Union[str, Dict[str, Any], List[Any]]


class ResponsesRequest(BaseModel):
    # Minimal subset of OpenAI Responses API
    model: str
    input: Optional[Union[str, List[Any]]] = None
    messages: Optional[List[ResponsesInputMessage]] = None  # Alternate to input
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None  # Prefer this if present
    max_tokens: Optional[int] = None  # Fallback if max_output_tokens not provided
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    stream: Optional[bool] = None  # We only support non-streaming
    tools: Optional[List[Any]] = None       # Accept tool definitions
    tool_choice: Optional[Any] = None

def _responses_to_chat_messages(req: ResponsesRequest) -> List[ChatMessage]:
    # Prefer explicit messages if provided
    if req.messages is not None:
        return [ChatMessage(role=m.role, content=m.content) for m in req.messages]

    # Otherwise, build a single user message from "input"
    if req.input is None:
        raise ValueError("Either 'messages' or 'input' must be provided")

    if isinstance(req.input, str):
        return [ChatMessage(role="user", content=req.input)]

    if isinstance(req.input, list):
        # Join list elements with newlines; stringify non-strings
        normalized = []
        for item in req.input:
            if isinstance(item, str):
                normalized.append(item)
            else:
                normalized.append(str(item))
        return [ChatMessage(role="user", content="\n".join(normalized))]

    raise ValueError("'input' must be a string or list")


def _to_iso8601(ts_seconds: int) -> str:
    """Helper to convert epoch seconds to ISO8601 string with 'Z' suffix --> for responses.created_at"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_seconds))


def _chat_to_responses(chat_obj: Dict[str, Any]) -> Dict[str, Any]:
    # Convert your OpenAI-like chat completion object to an OpenAI Responses-shaped object
    created_ts = chat_obj.get("created", int(time.time()))
    created_ms = int(created_ts * 1000)

    model = chat_obj.get("model", "")
    choices = chat_obj.get("choices", []) or []
    first = choices[0] if choices else {}
    message = first.get("message", {}) or {}
    content_text = (
        (message.get("content") or "")
        if isinstance(message.get("content"), str)
        else str(message.get("content") or "")
    )
    finish_reason = first.get("finish_reason", "stop")

    tool_calls = message.get("tool_calls")

    usage = chat_obj.get("usage", {}) or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    # ---- NEW: build output items based on whether tool_calls exist ----
    output_items = []

    if tool_calls:
        # Each tool call becomes a "function_call" output item (Responses API shape)
        for tc in tool_calls:
            func = tc.get("function", {})
            output_items.append({
                "id": tc.get("id", f"fc-{created_ms}"),
                "type": "function_call",
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "call_id": tc.get("id", f"call-{created_ms}"),
                "status": "completed",
            })
        stop_reason = "tool_use"
        output_text = content_text or ""
    else:
        # Normal text message
        output_items.append({
            "id": f"msg-{created_ms}",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": content_text},
                {"type": "text", "text": content_text},
            ],
        })
        stop_reason = "end_turn" if finish_reason in ("stop", None) else finish_reason
        output_text = content_text

    return {
        "id": f"resp-{created_ms}",
        "object": "response",
        "created": created_ts,
        "created_at": _to_iso8601(created_ts),
        "model": model,
        "status": "completed",
        "output": output_items,
        "output_text": output_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "metadata": {},
    }


@app.post("/v1/responses")
async def responses(req: ResponsesRequest):

    # Force streaming to False for now, since we don't support it
    if req.stream is True:
        logger.debug(
            "Streaming not supported in /v1/responses; forcing non streaming for now"
        )
        req.stream = False

    cfg_err = config_ok()
    if cfg_err is not None:
        return cfg_err

    # Build ChatCompletionRequest from ResponsesRequest
    try:
        msg_list = _responses_to_chat_messages(req)
    except ValueError as e:
        logger.debug(f"Invalid /v1/responses request: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    # Prefer max_output_tokens, fallback to max_tokens
    effective_max_tokens = (
        req.max_output_tokens if req.max_output_tokens is not None else req.max_tokens
    )

    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=msg_list,
        temperature=req.temperature,
        max_tokens=effective_max_tokens,
        stop=req.stop,
        seed=req.seed,
        response_format=req.response_format,
        stream=False,
        tools=[Tool(**t) if isinstance(t, dict) else t for t in req.tools] if req.tools else None,
        tool_choice=req.tool_choice,
    )

    # Build upstream payload identically to chat/completions
    try:
        payload = build_upstream_payload(chat_req)
    except ValueError as e:
        logger.debug(f"Invalid request (response_format): {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    upstream_url = f"{BASE_URL}/api/v1/llm/chat"
    headers = {
        "X-Client-ID": CLIENT_ID,
        "X-Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(upstream_url, headers=headers, json=payload)

            # If upstream NOT 2xx, synthesize OpenAI-like error
            if resp.status_code < 200 or resp.status_code >= 300:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text  # text is fallback again
                return JSONResponse(
                    status_code=resp.status_code,
                    content={
                        "error": {
                            "message": detail,
                            "type": "upstream_error",
                            "code": resp.status_code,
                        }
                    },
                )

            upstream_json = resp.json()
            logger.debug(f"Upstream response: {json.dumps(upstream_json, indent=2, default=str)}")

            # map to chat/completion shape an then to responses shape
            chat_like = map_upstream_to_openai_like(
                upstream_json, request_model=chat_req.model
            )
            responses_obj = _chat_to_responses(chat_like)
            return JSONResponse(status_code=200, content=responses_obj)

        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": f"Upstream timeout after {TIMEOUT_SECONDS} seconds",
                        "type": "upstream_error",
                        "code": 504,
                    }
                },
            )
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"HTTP error contacting upstream: {str(e)}",
                        "type": "upstream_error",
                        "code": 502,
                    }
                },
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
    # alternatively: uv run uvicorn app:app --host 0.0.0.0 --port 8081 --log-level debug