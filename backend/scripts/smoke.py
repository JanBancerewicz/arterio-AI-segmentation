"""HTTP smoke test against an already running Compose stack."""
import argparse
import getpass
import json
import warnings
from http.cookiejar import CookieJar
from io import BytesIO
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
from uuid import uuid4

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/api/v1"
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(path, data=None, headers=None):
        with opener.open(Request(base + path, data=data, headers=headers or {}), timeout=10) as response:
            return response.status, response.headers, response.read()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("Password: ")
    except (EOFError, getpass.GetPassWarning):
        parser.error("An interactive terminal is required to enter the password safely.")
    request("/auth/login", json.dumps({"username": args.username, "password": password}).encode(),
            {"Content-Type": "application/json"})
    del password
    _, _, health = request("/health")
    assert json.loads(health)["workersOnline"] >= 1, health
    image = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(image).line((20, 250, 490, 250), fill=255, width=12)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    boundary = str(uuid4())
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="modelId"\r\n\r\nsmoke-test\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
        'Content-Type: image/png\r\n\r\n'
    ).encode() + buffer.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    status, headers, data = request("/analyses", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 201
    analysis = json.loads(data)
    assert headers["Location"].endswith(analysis["id"])
    assert analysis["status"] == "queued"
    deadline = monotonic() + 30
    while monotonic() < deadline:
        _, _, data = request(f"/analyses/{analysis['id']}")
        analysis = json.loads(data)
        if analysis["status"] in ("completed", "failed"):
            break
        sleep(0.25)
    assert analysis["status"] == "completed", analysis
    for kind in ("original", "mask", "overlay"):
        _, headers, data = request(f"/analyses/{analysis['id']}/files/{kind}")
        assert headers.get_content_type() == "image/png"
        with Image.open(BytesIO(data)) as output:
            assert output.size == (512, 512)
            if kind == "mask":
                assert set(output.tobytes()) == {0, 255}
    try:
        request(f"/analyses/{analysis['id']}/cancel", b"")
        raise AssertionError("Completed job must not be cancellable")
    except HTTPError as exc:
        assert exc.code == 409
        assert json.load(exc)["error"]["code"] == "ANALYSIS_NOT_CANCELLABLE"
    _, _, data = request("/analyses?status=completed&pageSize=100")
    assert any(item["id"] == analysis["id"] for item in json.loads(data)["items"])
    request("/auth/logout", b"")
    print(json.dumps({"result": "PASS", "analysisId": analysis["id"], "processingMs": analysis["processingMs"]}))


if __name__ == "__main__":
    main()
