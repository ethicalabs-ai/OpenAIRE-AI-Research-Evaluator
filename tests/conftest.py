"""Shared pytest fixtures for the Echo-DSRN Graph test suite."""

import importlib
import os
import sys
import tempfile

import pytest

# Ensure backend directory is in path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
)


@pytest.fixture
def temp_db(monkeypatch):
    """
    Isolated SQLite database for task tests.

    Creates a fresh temp DB, runs migrations, and reloads the database
    module so all lazy imports inside Celery tasks resolve to the test DB.
    Cleans up the temp file on teardown.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_url = f"sqlite:///{tmp.name}"

    monkeypatch.setenv("DATABASE_URL", db_url)

    # Reload database.py so engine + SessionLocal point at the temp DB
    import database  # noqa: E402

    importlib.reload(database)

    # Create tables via Alembic (ensures schema matches production)
    import subprocess  # noqa: E402

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            os.path.join(repo_root, "backend", "alembic.ini"),
            "upgrade",
            "head",
        ],
        cwd=os.path.join(repo_root, "backend"),
        capture_output=True,
        check=True,
        env={**os.environ, "DATABASE_URL": db_url},
    )

    yield db_url

    os.unlink(tmp.name)


@pytest.fixture
def mock_judge_verdict():
    """Return a fake JudgeVerdict for happy-path tests."""
    from llm_judge import JudgeVerdict  # noqa: E402

    return JudgeVerdict(
        label_valid=True,
        proposed_label="Methodology",
        is_flagged=False,
        flag_reason=None,
        rationale="The paper introduces a novel architecture.",
        confidence="high",
    )


@pytest.fixture
def mock_classifier(monkeypatch):
    """Mock Echo-DSRN classifier to return 'Methodology' without loading the model."""
    from dataclasses import dataclass

    @dataclass
    class FakeResult:
        label: str = "Methodology"
        probabilities: dict = None  # noqa: RUF009

    def fake_classify(title, abstract):
        return FakeResult()

    monkeypatch.setattr(
        "intent_classifier.classify_paper",
        fake_classify,
        raising=False,
    )
