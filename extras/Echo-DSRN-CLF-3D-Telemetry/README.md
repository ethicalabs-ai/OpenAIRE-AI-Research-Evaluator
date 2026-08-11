---
title: Echo-DSRN CLF 3D Telemetry
emoji: 🌌
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
license: apache-2.0
pinned: false
short_description: 3D telemetry dashboard for the Echo-DSRN research-intent classifier on real OpenAIRE papers
---

# Echo-DSRN CLF 3D Telemetry

A 3D observational dashboard that peers into the recurrent core of the
**Echo-DSRN research-intent classifier** — adapted from the
[Echo-DSRN 114M Telemetry 3D](https://huggingface.co/spaces/ethicalabs/Echo-DSRN-114M-Telemetry-3D)
demo. Instead of typing free text, you pull **real research papers from the
OpenAIRE Graph API** (random paper, or by DOI) and watch the classifier's
neural dynamics as it reads the title + abstract.

### 🔗 Model
- **Research-intent classifier**: [ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF)
  (gated — set `HF_TOKEN` to download). Loaded through the AutoModel family
  with `trust_remote_code=True`.

---

## 🚀 What you can do

- **🎲 Pick a random paper** — the backend picks a random keyword and page from
  the OpenAIRE Graph API, returns a paper with a real title and abstract, then
  computes the per-token neural trace.
- **Enter a DOI** — lookup a specific publication (e.g. `10.1038/s41586-020-2649-2`),
  and trace its intent classification.
- **Classification telemetry** — every trace step shows the per-layer slow
  state (c_t), surprise gates (λ), fast state (h_t) bins and the classifier's
  *entropy core*: Shannon entropy over the class distribution as tokens
  accumulate. The final step's softmax probabilities drive the headline
  `PRED_INTENT` readout.

## 🎮 Interacting with the sandbox

1. Click **RANDOM PAPER** or type a DOI and hit the ➤ button.
2. The token trajectory auto-plays; pause and scrub the timeline to dissect
   spatial variance step-by-step.
3. While playback is suspended, raycast onto voxels, portals, the entropy core
   or filament splines to load diagnostic measurements into the HUD.

## 🧠 Architectural metaphors

| Structural Object | Neural Equivalency | Rendering Construct |
| :--- | :--- | :--- |
| **Input Block** | Token Embedding Stream | Subdivided Amber Pulsing Core |
| **λ-Gate Portals** | Surprise Gate Output ($\lambda$) | Interactive Hexagonal Iris Shutter |
| **Memory Stack** | Hidden Slow State Matrices ($c_t$) | Interactive 16-Grid Magma Voxels |
| **Filament Splines** | Attention Routing Network | Additive Raytraced Curved Splines |
| **Output Nexus** | Classification Head | Interactive Finalizing Cyan Tetrahedron |

## 🚀 Running

The model repo is **gated**: export a Hugging Face token that has access to
`ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF`.

```bash
docker build -t echo-clf-telemetry .
docker run -p 7860:7860 -e HF_TOKEN=hf_... echo-clf-telemetry
```

Locally (with your own venv):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend && npm ci && npm run build && cd ..
.venv/bin/python backend/server.py
```

Then open http://localhost:7860.

### Configuration

| Env var | Default | Purpose |
| :--- | :--- | :--- |
| `HF_TOKEN` | — | HF token for the gated classifier repo |
| `OPENAIRE_TOKEN` | — | Optional OpenAIRE API token (Bearer header) to raise rate limits; not needed for basic search |
| `VITE_FOOTER_TEXT` | `OpenAIRE Graph API data: CC-BY 4.0.` | Footer attribution line, inlined at frontend build time (`docker build --build-arg VITE_FOOTER_TEXT="..."`) |
| `ECHO_MAX_TRACE_TOKENS` | `1024` | Max tokens rendered in the per-token trace |
| `PORT` | `7860` | HTTP port |

*OpenAIRE Graph API data: CC-BY 4.0.*
