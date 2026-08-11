"""Tests for the Echo-DSRN-CLF-3D-Telemetry standalone app backend.

Covers the OpenAIRE result parsing (adapted from the evaluator's
judge_cli/server) and the classifier telemetry helpers (adapted from the
original 3D-telemetry server).  No network or model required — parsing is
fed raw OpenAIRE JSON samples and telemetry helpers get synthetic tensors.

Run from the app directory with its own venv:
    .venv/bin/pytest tests/ -q
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import openaire_client  # noqa: E402
import telemetry  # noqa: E402


# ── OpenAIRE parsing ──────────────────────────────────────────────────────────


def _result(
    title="A Study of Neural Networks",
    abstract="We propose a novel architecture <b>with markup</b>.\nSecond line.",
    doi=None,
    obj_id="od_______::abc123",
):
    pid = [{"@classid": "doi", "$": doi}] if doi else []
    return {
        "header": {"dri:objIdentifier": {"$": obj_id}},
        "metadata": {
            "oaf:entity": {
                "oaf:result": {
                    "title": [{"$": title}],
                    "description": [{"$": abstract}],
                    "creator": [{"$": "Alice"}, {"$": "Bob"}],
                    "pid": pid,
                    "instance": [{"webresource": {"url": {"$": "https://example.com/x"}}}],
                }
            }
        },
    }


def test_parse_result_extracts_clean_paper():
    paper = openaire_client.parse_result(_result(doi="https://doi.org/10.1000/abc"))
    assert paper is not None
    assert paper["title"] == "A Study of Neural Networks"
    assert paper["abstract"] == "We propose a novel architecture with markup. Second line."
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["doi"] == "10.1000/abc"  # doi.org prefix stripped
    assert paper["link"] == "https://doi.org/10.1000/abc"
    assert paper["source"] == "openaire"


def test_parse_result_skips_entries_without_title_or_abstract():
    assert openaire_client.parse_result(_result(title="   ")) is None
    assert openaire_client.parse_result(_result(abstract="")) is None
    assert openaire_client.parse_result("not a dict") is None


def test_parse_result_falls_back_to_openaire_id_when_no_doi():
    paper = openaire_client.parse_result(_result())
    assert paper["doi"] == "openaire-od_______::abc123"
    assert paper["link"] == "https://example.com/x"  # falls back to instance URL


def test_parse_results_handles_dict_and_list_shapes():
    single = _result(title="Only One Result")
    flat = openaire_client.parse_results(single)  # dict payload tolerated
    assert len(flat) == 1
    assert flat[0]["title"] == "Only One Result"

    # Single-field (dict, not list) title/description variants
    res = {
        "header": {"dri:objIdentifier": {"$": "od_______::x"}},
        "metadata": {
            "oaf:entity": {
                "oaf:result": {
                    "title": {"$": "Dict Title"},
                    "description": {"$": "A decent abstract here."},
                }
            }
        },
    }
    parsed = openaire_client.parse_results([res])
    assert parsed[0]["title"] == "Dict Title"
    assert parsed[0]["doi"].startswith("openaire-")


def test_fetch_by_doi_prefers_exact_match(monkeypatch):
    papers = [
        {"doi": "10.9999/other", "title": "Other", "abstract": "x" * 60},
        {"doi": "10.1000/abc", "title": "Target", "abstract": "y" * 60},
    ]
    monkeypatch.setattr(openaire_client, "_fetch", lambda params: papers)
    got = openaire_client.fetch_by_doi("https://doi.org/10.1000/abc")
    assert got["doi"] == "10.1000/abc"


def test_fetch_by_doi_empty_input():
    assert openaire_client.fetch_by_doi("  ") is None


def test_fetch_sends_bearer_header_when_token_set(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": {}})

    monkeypatch.setattr(openaire_client, "_OPENAIRE_TOKEN", "tok-123")
    monkeypatch.setattr(openaire_client.requests, "get", fake_get)
    openaire_client._fetch({"keywords": "deep learning"})
    assert captured["headers"].get("Authorization") == "Bearer tok-123"


# ── Telemetry helpers ─────────────────────────────────────────────────────────


def test_shannon_entropy_uniform_and_certain():
    uniform = torch.full((1, 6), 0.0)  # equal logits -> uniform distribution
    entropy = telemetry.shannon_entropy(uniform)
    assert entropy == pytest.approx(1.7918, abs=1e-3)  # ln(6)

    certain = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
    assert telemetry.shannon_entropy(certain) == pytest.approx(0.0, abs=1e-3)


def test_fast_state_bins_bin_averages():
    h = torch.arange(1, 33, dtype=torch.float32).unsqueeze(0)  # (1, 32)
    bins = telemetry.fast_state_bins(h, num_bins=16)
    assert len(bins) == 16
    assert bins[0] == pytest.approx((1 + 2) / 2)  # mean of first 2 values
    assert bins[-1] == pytest.approx((31 + 32) / 2)
    assert telemetry.fast_state_bins(None) == [0.0] * 16


def test_fast_state_bins_handles_sequence_dim():
    h = torch.ones(1, 5, 16)  # (B, T, D)
    bins = telemetry.fast_state_bins(h, num_bins=16)
    assert len(bins) == 16
    assert all(b == pytest.approx(1.0) for b in bins)  # abs of last timestep


def test_attention_links_scale_with_lambda():
    assert telemetry.attention_links(0.0) == []
    assert telemetry.attention_links(0.05) == []  # not > 0.05
    assert len(telemetry.attention_links(0.5)) == 2  # int(0.5 * 5)
    assert len(telemetry.attention_links(0.9)) == 3  # capped at 3
    assert all(link["val"] == 0.9 for link in telemetry.attention_links(0.9))


def test_probabilities_from_logits_uses_id2label():
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "Methodology", 1: "Dataset"})
    )
    logits = torch.tensor([[2.0, 0.0]])
    probs = telemetry.probabilities_from_logits(model, logits)
    assert set(probs) == {"Methodology", "Dataset"}
    assert probs["Methodology"] == pytest.approx(0.8808, abs=1e-3)
    assert probs["Dataset"] == pytest.approx(0.1192, abs=1e-3)


def test_format_model_text_no_templates_returns_raw():
    model = SimpleNamespace(config=SimpleNamespace(system_prompt=None, user_template=None))
    text = "Title: X\nAbstract: Y"
    assert telemetry.format_model_text(model, None, text) == text


def test_format_model_text_template_failure_falls_back_to_raw():
    model = SimpleNamespace(
        config=SimpleNamespace(
            system_prompt="sys", user_template="Q: {text}", id2label={}
        )
    )
    class NoChat:
        pass

    assert telemetry.format_model_text(model, NoChat(), "hello") == "hello"


def test_classifier_logits_pools_last_token():
    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    model = SimpleNamespace(
        dropout=torch.nn.Identity(),
        classifier=FakeClassifier(),
    )
    torch.manual_seed(0)
    hidden = torch.randn(1, 7, 4)
    logits = telemetry.classifier_logits(model, hidden)
    assert logits.shape == (1, 3)
