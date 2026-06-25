import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "echo_dsrn_graph", broker=REDIS_URL, backend=REDIS_URL, include=["tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Workload Concurrency Division Routing
celery_app.conf.task_routes = {
    "tasks.ping_task": {"queue": "cpu"},
    "tasks.classify_and_judge": {"queue": "cpu"},
    "tasks.classify_mcp": {"queue": "cpu"},
}
