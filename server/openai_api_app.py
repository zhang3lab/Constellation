from __future__ import annotations

import json
import threading
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server.chat_completions_adapter import run_chat_completions
from server.openai_api_runtime import OpenAIAPIRuntime


app = FastAPI()
RUNTIME = OpenAIAPIRuntime(config_path="server/test/config.json")
_REQUEST_LOCK = threading.Lock()


@app.on_event("startup")
def startup() -> None:
    RUNTIME.start()


@app.on_event("shutdown")
def shutdown() -> None:
    RUNTIME.close()


def get_active_session():
    return RUNTIME.get_session()


def make_openai_error_payload(
    *,
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> dict:
    if not isinstance(message, str) or not message:
        raise TypeError("message must be non-empty str")
    if not isinstance(error_type, str) or not error_type:
        raise TypeError("error_type must be non-empty str")
    if param is not None and not isinstance(param, str):
        raise TypeError("param must be str|None")
    if code is not None and not isinstance(code, str):
        raise TypeError("code must be str|None")

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def make_openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=int(status_code),
        content=make_openai_error_payload(
            message=message,
            error_type=error_type,
            param=param,
            code=code,
        ),
    )


def _sse_encode(obj) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def _split_text_chunks(text: str, chunk_size: int = 16) -> list[str]:
    if not isinstance(text, str):
        raise TypeError(f"text expected str, got {type(text).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def iter_sse_from_chat_result(result: dict) -> Iterator[bytes]:
    if not isinstance(result, dict):
        raise TypeError(f"result expected dict, got {type(result).__name__}")

    result_id = result.get("id")
    result_object = result.get("object")
    created = result.get("created")
    model = result.get("model")
    choices = result.get("choices")

    if not isinstance(result_id, str) or not result_id:
        raise RuntimeError("result['id'] expected non-empty str")
    if result_object != "chat.completion":
        raise RuntimeError(f"result['object'] expected 'chat.completion', got {result_object!r}")
    if not isinstance(created, int):
        raise RuntimeError(f"result['created'] expected int, got {type(created).__name__}")
    if not isinstance(model, str) or not model:
        raise RuntimeError("result['model'] expected non-empty str")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("result['choices'] expected single-choice list")

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise RuntimeError("result['choices'][0] expected dict")

    index = choice0.get("index", 0)
    message = choice0.get("message")
    finish_reason = choice0.get("finish_reason")

    if not isinstance(index, int):
        raise RuntimeError("result['choices'][0]['index'] expected int")
    if not isinstance(message, dict):
        raise RuntimeError("result['choices'][0]['message'] expected dict")

    role = message.get("role")
    content = message.get("content", "")

    if role != "assistant":
        raise RuntimeError(f"result['choices'][0]['message']['role'] expected 'assistant', got {role!r}")
    if not isinstance(content, str):
        raise RuntimeError("result['choices'][0]['message']['content'] expected str")

    base = {
        "id": result_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }

    yield _sse_encode(
        {
            **base,
            "choices": [
                {
                    "index": index,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
            ],
        }
    )

    for part in _split_text_chunks(content, chunk_size=16):
        yield _sse_encode(
            {
                **base,
                "choices": [
                    {
                        "index": index,
                        "delta": {
                            "content": part,
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield _sse_encode(
        {
            **base,
            "choices": [
                {
                    "index": index,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )

    yield _sse_done()


@app.get("/healthz")
def healthz():
    try:
        session = get_active_session()
    except Exception as e:
        return make_openai_error_response(
            status_code=500,
            message=f"session unavailable: {e}",
            error_type="server_error",
            code="session_unavailable",
        )

    return {
        "ok": True,
        "chat_runtime_ready": bool(session.is_chat_runtime_ready()),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req = await request.json()
    except Exception as e:
        return make_openai_error_response(
            status_code=400,
            message=f"invalid json: {e}",
            error_type="invalid_request_error",
            code="invalid_json",
        )

    if not isinstance(req, dict):
        return make_openai_error_response(
            status_code=400,
            message=f"request body must be json object, got {type(req).__name__}",
            error_type="invalid_request_error",
            code="invalid_request_body",
        )

    try:
        session = get_active_session()
    except Exception as e:
        return make_openai_error_response(
            status_code=500,
            message=f"failed to get session: {e}",
            error_type="server_error",
            code="session_unavailable",
        )

    try:
        with _REQUEST_LOCK:
            resp = run_chat_completions(
                session,
                request=req,
                return_aux=False,
            )
    except TypeError as e:
        return make_openai_error_response(
            status_code=400,
            message=str(e),
            error_type="invalid_request_error",
            code="type_error",
        )
    except ValueError as e:
        return make_openai_error_response(
            status_code=400,
            message=str(e),
            error_type="invalid_request_error",
            code="value_error",
        )
    except RuntimeError as e:
        # 当前先统一按 invalid_request_error 处理；
        # 以后如果你把 adapter 的请求错误和内部错误分开，可以再细分 400/500
        return make_openai_error_response(
            status_code=400,
            message=str(e),
            error_type="invalid_request_error",
            code="runtime_error",
        )
    except Exception as e:
        return make_openai_error_response(
            status_code=500,
            message=f"internal error: {e}",
            error_type="server_error",
            code="internal_error",
        )

    if not isinstance(resp, dict):
        return make_openai_error_response(
            status_code=500,
            message=f"adapter returned non-dict response: {type(resp).__name__}",
            error_type="server_error",
            code="invalid_adapter_response",
        )

    result = resp.get("result")
    stream = resp.get("stream", False)

    if not isinstance(result, dict):
        return make_openai_error_response(
            status_code=500,
            message="adapter response missing dict field 'result'",
            error_type="server_error",
            code="invalid_adapter_response",
        )

    if not isinstance(stream, bool):
        return make_openai_error_response(
            status_code=500,
            message="adapter response field 'stream' must be bool",
            error_type="server_error",
            code="invalid_adapter_response",
        )

    if not stream:
        return JSONResponse(content=result)

    try:
        return StreamingResponse(
            iter_sse_from_chat_result(result),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        return make_openai_error_response(
            status_code=500,
            message=f"failed to start stream: {e}",
            error_type="server_error",
            code="stream_init_failed",
        )
