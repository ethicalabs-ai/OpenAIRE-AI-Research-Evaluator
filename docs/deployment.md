# Kubernetes Deployment (k3s)

Deploy the OpenAIRE-AI-Research-Evaluator with Helm on k3s.

## Prerequisites

- k3s cluster running
- `helm` installed
- Access to the Docker image `ethicalabs/openaire-research-evaluator:latest`
- The Echo-DSRN model (public on HuggingFace, baked into the image at build)

## 1. Install

```bash
helm install evaluator ./helm/openaire-research-evaluator \
  --set secrets.hfToken=hf_... \
  --set secrets.secretKey=$(openssl rand -hex 32) \
  --set ingress.host=openaire-eval.local \
  --set config.llmBaseUrl=http://<llm-server-host>:<llm-server-port>/v1
```

Or create a `values.local.yaml`:

```yaml
secrets:
  hfToken: hf_...
  secretKey: abc123...
  hfClientId: ...
  hfClientSecret: ...
ingress:
  host: openaire-eval.local
config:
  llmBaseUrl: http://<llm-server-host>:<llm-server-port>/v1
  # Active collab annotation round (stamped on new papers/annotations).
  modelVersion: "v0.1.4"
  # Set true to close the hub (browse-only). Default false.
  hubReadOnly: false
```

```bash
helm install evaluator ./helm/openaire-research-evaluator -f values.local.yaml
```

The web container runs `alembic upgrade head` on startup — existing rows are
backfilled to the legacy round (`v0.1.3`) and stay browsable at
`/collab/archive/v0.1.3/`.

## 2. Access

```bash
kubectl port-forward svc/evaluator-openaire-research-evaluator-web 7860:7860
```

App at `http://localhost:7860`. With Ingress configured, at `http://openaire-eval.local`.

**Local DNS setup:** add to `/etc/hosts`:

```bash
echo "127.0.0.1 openaire-eval.local" | sudo tee -a /etc/hosts
```

Then access at `http://openaire-eval.local` (k3s Traefik listens on port 80).

## Components

| Component | Service | Port |
|---|---|---|
| Web (FastAPI + Vue) | `<release>-web` | 7860 |
| PostgreSQL | `<release>-postgres` | 5432 |
| Redis | `<release>-redis` | 6379 |
| Worker (Celery) | — (internal) | — |

## Troubleshooting

Check web pod logs for alembic migration status:

```bash
kubectl logs deployment/evaluator-openaire-research-evaluator-web
```
