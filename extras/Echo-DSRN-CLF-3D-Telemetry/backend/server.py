from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import openaire_client
import telemetry

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)

MODEL_ID = "ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF"

# Global model/tokenizer
tokenizer = None
model = None
model_loaded = False


def load_model():
    global tokenizer, model, model_loaded
    print(f"Loading classifier {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # AutoModel family — trust_remote_code loads the Echo classes from the
        # model repo and registers them with HF AutoClass routing.
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        model_loaded = True
        print(f"Model loaded successfully. Device: {device}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model_loaded = False


def _paper_to_trace_input(paper: dict) -> str:
    """Compose the classifier input the same way the evaluator does."""
    return f"Title: {paper['title']}\nAbstract: {paper['abstract']}"


@app.route("/api/trace", methods=["POST"])
def get_trace():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Backend may still be initializing."}), 503

    try:
        data = request.json or {}
        paper = data.get("paper") or {}
        title = paper.get("title", "") or data.get("title", "")
        abstract = paper.get("abstract", "") or data.get("abstract", "")
        if not title or not abstract:
            return jsonify({"error": "A paper title and abstract are required"}), 400

        text = _paper_to_trace_input({"title": title, "abstract": abstract})
        trace, prediction = telemetry.compute_trace(model, tokenizer, text)
        return jsonify(
            {
                "trace": trace,
                "prediction": prediction,
                "paper": {"title": title, "abstract": abstract},
            }
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            return (
                jsonify({"error": "Out of memory. Restart the backend or shorten the abstract."}),
                503,
            )
        return jsonify({"error": f"Runtime error: {error_msg[:200]}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)[:200]}"}), 500


@app.route("/api/openaire/random")
def openaire_random():
    """Return metadata for a pseudo-random OpenAIRE paper."""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        paper = openaire_client.fetch_random_paper()
        if not paper:
            return jsonify({"error": "No paper found — OpenAIRE unreachable or empty result."}), 502
        return jsonify(paper)
    except Exception as e:
        return jsonify({"error": f"OpenAIRE fetch failed: {str(e)[:200]}"}), 500


@app.route("/api/openaire/search")
def openaire_search():
    """Return a publication looked up by DOI."""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503
    doi = request.args.get("doi", "").strip()
    if not doi:
        return jsonify({"error": "Missing doi query parameter"}), 400
    try:
        paper = openaire_client.fetch_by_doi(doi)
        if not paper:
            return jsonify({"error": f"No paper found for DOI {doi!r}."}), 404
        return jsonify(paper)
    except Exception as e:
        return jsonify({"error": f"OpenAIRE fetch failed: {str(e)[:200]}"}), 500


@app.route("/api/config")
def get_config():
    if not model_loaded:
        return jsonify({"error": "Model not loaded", "layers": 0}), 503
    return jsonify(
        {
            "layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "heads": model.config.num_heads,
            "state_dim": model.config.hidden_size * model.config.num_heads,
            "labels": list(model.config.id2label.values()),
            "model_id": MODEL_ID,
        }
    )


@app.route("/api/health")
def health_check():
    if not model_loaded:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    load_model()
    import os

    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
