import os
import time
from typing import List, Optional, Union, Dict, Any

import json
import re

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
TIMEOUT_SECONDS = float(os.environ.get("TIMEOUT_SECONDS", "90"))

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

app = FastAPI(title="OpenAI-compatible shim for AcademicAI API", version="0.1.0")

# ==================================================
# ================Shitty tool calling===============
# ==================================================


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


# Register your tools here: name -> (callable, json-schema-like params)
# Each callable must accept **kwargs and return a serializable dict or string.
def tool_write(file_path: str, content: str) -> Dict[str, Any]:
    # EXAMPLE TOOL: Writes a file. Adjust path safety for your environment!

    logger.info(f"Tool Write called with file_path={file_path}, content length={len(content)}")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "file_path": file_path, "bytes": len(content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Write": {
        "callable": tool_write,
        "schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to write",
                },
                "content": {"type": "string", "description": "File contents"},
            },
            "required": ["file_path", "content"],
            "additionalProperties": False,
        },
        "description": "Writes a file to the local filesystem.",
    },
    # Add more tools here...
}

# The protocol we’ll instruct the model to use when it wants a tool:
# It must emit a single fenced JSON block:
# ```tool
# {"name": "Write", "arguments": {"file_path": "...", "content": "..."}}
# ```
TOOL_CALL_PATTERN = re.compile(r"```tool\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def build_tools_system_prompt(tools: Optional[List[Tool]]) -> str:
    # Summarize available tools with names and schemas
    if not tools:
        return ""
    lines = [
        "You can use the following tools by requesting them in a single fenced JSON block.",
        "When you want to call a tool, respond ONLY with:",
        "```tool",
        '{"name": "<TOOL_NAME>", "arguments": { ... }}',
        "```",
        "Rules:",
        "- Do not include any other text outside the fenced block when calling a tool.",
        "- arguments must be valid JSON.",
        "- After receiving the tool result, continue your answer normally.",
        "",
        "Available tools:",
    ]
    for t in tools:
        f = t.function
        schema = f.parameters or {"type": "object", "properties": {}}
        lines.append(f"- {f.name}: {f.description or ''}")
        lines.append(f"  schema: {json.dumps(schema, ensure_ascii=False)}")
    return "\n".join(lines)


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls as [{'name': str, 'arguments': dict}]."""
    if not text:
        return []
    calls = []
    for m in TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name")
            arguments = payload.get("arguments") or {}
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
        except Exception:
            continue
    return calls


def execute_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return a result dict."""
    entry = TOOL_REGISTRY.get(name)

    if not entry:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    fn = entry["callable"]
    try:
        result = fn(**arguments)
        # Ensure it is JSON-serializable
        if isinstance(result, (str, int, float, bool)) or result is None:
            return {"ok": True, "result": result}
        return result if isinstance(result, dict) else {"ok": True, "result": result}
    except TypeError as e:
        return {"ok": False, "error": f"Invalid arguments: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class ChatMessage(BaseModel):
    role: str
    content: Union[str, Dict[str, Any], List[Any]]


class ResponseFormat(BaseModel):
    # Only "json_object" or "text" allowed; validated in build_upstream_payload
    type: Optional[str] = None


# ==================================================
# ============== Content normalization =============
# ==================================================

def _content_to_text(content: Union[str, Dict[str, Any], List[Any], None]) -> str:
    """
    Normalize any message content into a plain text string for the upstream API.

    Rules:
    - If it's a string -> return as-is.
    - If it's a list -> try to concatenate any 'text' fields from dict items; otherwise JSON-serialize items.
    - If it's a dict -> prefer dict.get('text') if present; otherwise JSON-serialize.
    - None -> empty string.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # Anthropic/OpenAI Responses often use a list of blocks
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                # common shapes: {"type":"text","text":"..."} or {"text":"..."}
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                else:
                    # last resort: serialize the dict
                    try:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    except Exception:
                        parts.append(str(item))
            else:
                # primitive or unknown -> stringify
                parts.append(str(item))
        return "\n".join(p for p in parts if p)

    if isinstance(content, dict):
        # Prefer a top-level 'text' field if present
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    # Fallback
    return str(content)


def _normalize_messages_for_upstream(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Return [{'role': str, 'content': str}, ...] with:
    - content normalized to text
    - role coerced to 'user' if not 'user' or 'assistant'
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.role
        if role not in ("user", "assistant"):
            logger.debug(f"Coercing unsupported role '{role}' to 'user' for upstream")
            role = "user"
        out.append({"role": role, "content": _content_to_text(m.content)})
    return out


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
    model_config = {"extra": "allow"}


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
        "messages": _normalize_messages_for_upstream(req.messages),
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
        payload["toolChoice"] = req.tool_choice  # keep casing aligned with local usage

    return payload


async def upstream_chat_once(
    client: httpx.AsyncClient,
    upstream_url: str,
    headers: Dict[str, str],
    messages: List[ChatMessage],
    temperature: Optional[float],
    max_tokens: Optional[int],
    response_format: Optional[ResponseFormat],
    stop: Optional[Union[str, List[str]]],
    seed: Optional[int],
) -> Dict[str, Any]:
    """Send one non-streaming turn to the upstream API and return upstream JSON."""
    payload = {
        "model": "",  # the caller will set model; kept for shape
        "messages": _normalize_messages_for_upstream(messages),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["maxTokens"] = max_tokens
    if stop is not None:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed
    if response_format and response_format.type:
        payload["responseFormat"] = {"type": response_format.type}

    resp = await client.post(upstream_url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


async def run_with_local_tools(
    req_model: str,
    base_messages: List[ChatMessage],
    tools: Optional[List[Tool]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    response_format: Optional[ResponseFormat],
    stop: Optional[Union[str, List[str]]],
    seed: Optional[int],
    max_tool_iterations: int = 4,
) -> Dict[str, Any]:
    """
    Orchestrate tool use:
      1) Inject a system instruction describing the tool-call protocol and available tools.
      2) Send to upstream; parse any tool call from assistant text.
      3) Execute tool(s) locally; append results as a system message.
      4) Repeat until model emits no tool call or iteration cap reached.
    Returns either:
      - {"final_upstream": upstream_json, "tool_calls": [...]} on success
      - {"error": {"status": int, "body": Any}} on upstream failure
    """
    upstream_url = f"{BASE_URL}/api/v1/llm/chat"
    headers = {
        "X-Client-ID": CLIENT_ID,
        "X-Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json",
    }

    # Build messages with tool system prompt
    tool_sys = build_tools_system_prompt(tools) if tools else ""
    messages = list(base_messages)
    if tools and tool_sys:
        messages = [ChatMessage(role="system", content=tool_sys)] + messages

    tool_calls_out: List[Dict[str, Any]] = []

    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for iteration in range(max_tool_iterations + 1):
            # Call upstream once with normalized messages
            payload = {
                "model": req_model,
                "messages": _normalize_messages_for_upstream(messages),
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["maxTokens"] = max_tokens
            if stop is not None:
                payload["stop"] = stop
            if seed is not None:
                payload["seed"] = seed
            if response_format and response_format.type:
                payload["responseFormat"] = {"type": response_format.type}

            # Ensure no tool metadata is forwarded upstream
            payload.pop("tools", None)
            payload.pop("toolChoice", None)

            try:
                resp = await client.post(upstream_url, headers=headers, json=payload)
            except httpx.TimeoutException:
                return {"error": {"status": 504, "body": f"Upstream timeout after {TIMEOUT_SECONDS} seconds"}}
            except httpx.HTTPError as e:
                return {"error": {"status": 502, "body": f"HTTP error contacting upstream: {str(e)}"}}

            if resp.status_code < 200 or resp.status_code >= 300:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                logger.error(f"Upstream non-2xx in tool loop: {resp.status_code} - {detail}")
                return {"error": {"status": resp.status_code, "body": detail}}

            upstream_json = resp.json()
            data = upstream_json.get("data") or {}
            assistant_text = data.get("content") or ""
            # Try to parse tool calls from assistant_text
            calls = parse_tool_calls_from_text(assistant_text)

            if not calls:
                # No tool call: final answer
                return {
                    "final_upstream": upstream_json,
                    "tool_calls": tool_calls_out,
                }

            # Append the assistant turn that made the tool request first
            messages.append(ChatMessage(role="assistant", content=assistant_text))

            # Execute each call and append results; also track for OpenAI-like output
            created_ms = int(time.time() * 1000)
            for i, call in enumerate(calls):
                name = call["name"]
                args = call["arguments"]

                logger.info(f"Executing tool call {i+1}/{len(calls)}: {name} with args={args}")

                result = execute_tool_call(name, args)

                # Record in OpenAI-style for later mapping
                tool_calls_out.append(
                    {
                        "id": f"call_{created_ms}_{i}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                        "result": result,  # keep locally; not part of OpenAI tool_calls, but useful
                    }
                )

                # Feed result back to the model as a system message
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"Tool result for {name}: {json.dumps(result, ensure_ascii=False)}",
                    )
                )

        # Iteration cap reached
        return {
            "final_upstream": upstream_json
            if "upstream_json" in locals()
            else {
                "data": {
                    "model": req_model,
                    "role": "assistant",
                    "content": "Tool iteration limit reached.",
                    "finishReason": "stop",
                }
            },
            "tool_calls": tool_calls_out,
        }


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

    # Parse any upstream-declared tool calls (if your upstream ever provides them)
    raw_tool_calls = data.get("toolCalls") or data.get("tool_calls")

    message: Dict[str, Any] = {
        "role": role,
        "content": content,  # can be None when tool_calls are present
    }

    if raw_tool_calls:
        # Normalize to OpenAI tool_calls format
        tool_calls = []
        for i, tc in enumerate(raw_tool_calls):
            func = tc.get("function") or {}
            tool_calls.append(
                {
                    "id": tc.get("id") or f"call_{created_ms}_{i}",
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),  # JSON string
                    },
                }
            )
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None
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
    if req.stream is True:
        logger.debug(
            "Streaming not supported in /v1/chat/completions; forcing non-streaming for now"
        )
        req.stream = False

    cfg_err = config_ok()
    if cfg_err is not None:
        return cfg_err

    # If tools are provided and tool_choice is not "none", run local tool loop
    use_tools = bool(req.tools) and (req.tool_choice != "none")

    if use_tools:
        try:
            result = await run_with_local_tools(
                req_model=req.model,
                base_messages=req.messages,
                tools=req.tools,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                response_format=req.response_format,
                stop=req.stop,
                seed=req.seed,
            )

            # Handle upstream/tool-loop error
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                return JSONResponse(
                    status_code=err.get("status", 500),
                    content={
                        "error": {
                            "message": err.get("body"),
                            "type": "upstream_error",
                            "code": err.get("status", 500),
                        }
                    },
                )

            upstream_json = result["final_upstream"]
            tool_calls = result["tool_calls"]

            # Map upstream to OpenAI-like
            chat_like = map_upstream_to_openai_like(
                upstream_json, request_model=req.model
            )

            # Inject tool_calls into the first choice message if we had any
            if tool_calls:
                msg = chat_like["choices"][0]["message"]
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in tool_calls
                ]
                # If model’s text was only the tool block, set content None like OpenAI
                if isinstance(msg.get("content"), str) and TOOL_CALL_PATTERN.search(
                    msg["content"]
                ):
                    msg["content"] = None
                # Set finish_reason to "tool_calls" to mirror OpenAI
                chat_like["choices"][0]["finish_reason"] = "tool_calls"

            return JSONResponse(status_code=200, content=chat_like)
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

    # No tools: fall back to your existing simple path
    try:
        payload = build_upstream_payload(req)
    except ValueError as e:
        logger.debug(f"Invalid request: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    # IMPORTANT: strip tools/tool_choice from payload for this upstream
    payload.pop("tools", None)
    payload.pop("toolChoice", None)

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
            if resp.status_code < 200 or resp.status_code >= 300:
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
    tools: Optional[List[Any]] = None  # Accept tool definitions
    tool_choice: Optional[Any] = None
    model_config = {"extra": "allow"}


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

    # Build output items based on whether tool_calls exist
    output_items = []

    if tool_calls:
        # Each tool call becomes a "function_call" output item (Responses API shape)
        for tc in tool_calls:
            func = tc.get("function", {})
            output_items.append(
                {
                    "id": tc.get("id", f"fc-{created_ms}"),
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                    "call_id": tc.get("id", f"call-{created_ms}"),
                    "status": "completed",
                }
            )
        stop_reason = "tool_use"
        output_text = content_text or ""
    else:
        # Normal text message
        output_items.append(
            {
                "id": f"msg-{created_ms}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content_text},
                    {"type": "text", "text": content_text},
                ],
            }
        )
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
    if req.stream is True:
        logger.debug("Streaming not supported in /v1/responses; forcing non streaming for now")
        req.stream = False

    cfg_err = config_ok()
    if cfg_err is not None:
        return cfg_err

    try:
        msg_list = _responses_to_chat_messages(req)
    except ValueError as e:
        logger.debug(f"Invalid /v1/responses request: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    effective_max_tokens = (
        req.max_output_tokens if req.max_output_tokens is not None else req.max_tokens
    )

    # Normalize tools into Tool models if they exist
    my_tools = None
    if req.tools:
        my_tools = []
        for t in req.tools:
            if isinstance(t, dict) and "function" in t:
                my_tools.append(Tool(**t))
            elif isinstance(t, dict) and "name" in t:
                my_tools.append(
                    Tool(
                        type=t.get("type", "function"),
                        function=ToolFunction(
                            name=t["name"],
                            description=t.get("description"),
                            parameters=t.get("parameters"),
                        ),
                    )
                )

    use_tools = bool(my_tools) and (req.tool_choice != "none")

    if use_tools:
        # Run local tool loop; do not forward tools upstream
        result = await run_with_local_tools(
            req_model=req.model,
            base_messages=msg_list,
            tools=my_tools,
            temperature=req.temperature,
            max_tokens=effective_max_tokens,
            response_format=req.response_format,
            stop=req.stop,
            seed=req.seed,
        )

        # If upstream returned an error during the loop, pass it through
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            return JSONResponse(
                status_code=err.get("status", 500),
                content={
                    "error": {
                        "message": err.get("body"),
                        "type": "upstream_error",
                        "code": err.get("status", 500),
                    }
                },
            )

        upstream_json = result["final_upstream"]
        tool_calls = result["tool_calls"]

        # Map to OpenAI chat-like and inject tool_calls so we can convert to Responses
        chat_like = map_upstream_to_openai_like(upstream_json, request_model=req.model)
        if tool_calls:
            msg = chat_like["choices"][0]["message"]
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": tc["function"]}
                for tc in tool_calls
            ]
            # If the assistant text only contained the fenced tool JSON, set content to None
            if isinstance(msg.get("content"), str) and TOOL_CALL_PATTERN.search(msg["content"]):
                msg["content"] = None
            chat_like["choices"][0]["finish_reason"] = "tool_calls"

        responses_obj = _chat_to_responses(chat_like)
        return JSONResponse(status_code=200, content=responses_obj)

    # ---------- No tools: fall back to the upstream path ----------
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=msg_list,
        temperature=req.temperature,
        max_tokens=effective_max_tokens,
        stop=req.stop,
        seed=req.seed,
        response_format=req.response_format,
        stream=False,
        tools=None,
        tool_choice=None,
    )

    try:
        payload = build_upstream_payload(chat_req)
    except ValueError as e:
        logger.debug(f"Invalid request (response_format): {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "bad_request", "code": 400}},
        )

    # Ensure unsupported fields are not forwarded
    payload.pop("tools", None)
    payload.pop("toolChoice", None)

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
            if resp.status_code < 200 or resp.status_code >= 300:
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