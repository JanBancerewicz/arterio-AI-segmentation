"""Load test of the API: locust -f perf/locustfile.py --host http://localhost:8000"""
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
from time import monotonic, sleep
from uuid import uuid4

from locust import between, events, HttpUser, task
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_SCRIPT = """
import json, secrets, sys
from app import sessions
from app.users import create_user
tokens = []
for index in range(int(sys.argv[1])):
    username, password = f"perf.{sys.argv[2]}.{index}", secrets.token_urlsafe(24)
    create_user(username, "Test wydajnosci", password)
    tokens.append(sessions.login(username, password).token)
print(json.dumps(tokens))
"""
tokens = []


def compose(*args):
    return subprocess.run(["docker", "compose", *args], cwd=PROJECT_ROOT, check=True,
                          capture_output=True, text=True).stdout


@events.test_start.add_listener
def create_sessions(environment, **_):
    users = environment.parsed_options.num_users if environment.parsed_options else 10
    output = compose("exec", "-T", "api", "python", "-c", SESSION_SCRIPT, str(users), uuid4().hex[:8])
    tokens.extend(json.loads(output.strip().splitlines()[-1]))


def frame():
    image = Image.new("L", (512, 512), 30)
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        draw.line([(random.randint(0, 511), random.randint(0, 511)) for _ in range(2)], fill=230, width=10)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class SignedInUser(HttpUser):
    abstract = True

    def on_start(self):
        self.client.headers["Cookie"] = f"arterio_session={tokens.pop()}"
        self.analysis_ids = []


class ReviewingUser(SignedInUser):
    weight = 3
    wait_time = between(1, 3)

    @task(4)
    def history(self):
        response = self.client.get("/api/v1/analyses?pageSize=20", name="GET /analyses")
        if response.ok:
            self.analysis_ids = [item["id"] for item in response.json()["items"]]

    @task(2)
    def details(self):
        if self.analysis_ids:
            self.client.get(f"/api/v1/analyses/{random.choice(self.analysis_ids)}", name="GET /analyses/{id}")

    @task(1)
    def session(self):
        self.client.get("/api/v1/auth/me", name="GET /auth/me")

    @task(1)
    def models(self):
        self.client.get("/api/v1/models", name="GET /models")

    @task(1)
    def upload_and_preview(self):
        response = self.client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                                    files={"file": ("perf.png", frame(), "image/png")}, name="POST /analyses")
        if response.status_code == 201:
            analysis_id = response.json()["id"]
            self.analysis_ids.append(analysis_id)
            self.client.get(f"/api/v1/analyses/{analysis_id}/files/original", name="GET /files/original")


class SegmentingUser(SignedInUser):
    weight = 1
    wait_time = between(2, 5)

    @task
    def segment(self):
        started = monotonic()
        response = self.client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                                    files={"file": ("perf.png", frame(), "image/png")}, name="POST /analyses")
        if response.status_code != 201:
            return
        analysis = response.json()
        while analysis["status"] in ("queued", "processing") and monotonic() - started < 120:
            sleep(0.25)
            analysis = self.client.get(f"/api/v1/analyses/{analysis['id']}", name="GET /analyses/{id} (polling)").json()
        events.request.fire(
            request_type="E2E", name="upload -> gotowa maska", response_time=(monotonic() - started) * 1000,
            response_length=0, context={},
            exception=None if analysis["status"] == "completed" else RuntimeError(analysis["status"]),
        )
        if analysis["status"] == "completed":
            self.client.get(f"/api/v1/analyses/{analysis['id']}/files/mask", name="GET /files/mask")
