"""Errors rendered in OpenAI's error envelope.

`openai` SDK clients read `{"error": {"message", "type", "param", "code"}}` and
raise a typed exception (`BadRequestError`) with the message intact, so every
4xx we produce carries that shape instead of FastAPI's `{"detail": ...}`.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class OpenAIError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error",
                 code: str = None, param: str = None):
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param


def envelope(status: int, message: str, err_type: str = "invalid_request_error",
             code: str = None, param: str = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "param": param, "code": code}},
        status_code=status,
    )


def install(app: FastAPI) -> None:
    @app.exception_handler(OpenAIError)
    async def _openai_error(_: Request, exc: OpenAIError):
        return envelope(exc.status, exc.message, exc.err_type, exc.code, exc.param)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "validation failed")
        if loc:
            message = f"invalid request: {loc}: {message}"
        return envelope(400, message, "invalid_request_error", "invalid_request")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return envelope(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        # Never leak a traceback body to an API client; Railway logs keep it.
        return envelope(500, f"internal error: {type(exc).__name__}", "server_error")
