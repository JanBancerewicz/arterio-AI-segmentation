"""Bound request bytes before multipart parsing can spool an oversized file."""
from uuid import uuid4

from starlette.responses import JSONResponse

from . import config


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await self.app(scope, receive, send)
        limit = config.MAX_LOGIN_BYTES if scope["path"] == "/api/v1/auth/login" else config.MAX_REQUEST_BYTES

        async def reject(status, code, message):
            response = JSONResponse(status_code=status, content={"error": {
                "code": code, "message": message, "details": {"maxRequestBytes": limit},
                "requestId": scope.get("state", {}).get("request_id", str(uuid4())),
            }})
            await response(scope, receive, send)

        length = dict(scope.get("headers", [])).get(b"content-length")
        if length is not None:
            try:
                length = int(length)
                if length < 0:
                    raise ValueError()
            except ValueError:
                return await reject(400, "VALIDATION_ERROR", "Niepoprawny nagłówek Content-Length.")
            if length > limit:
                return await reject(413, "REQUEST_TOO_LARGE", "Żądanie przekracza dozwolony rozmiar.")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                return await reject(413, "REQUEST_TOO_LARGE", "Żądanie przekracza dozwolony rozmiar.")
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        data = bytes(body)
        del body
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": data, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
