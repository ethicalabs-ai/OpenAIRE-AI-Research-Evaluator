# Kubernetes Deployment (k3s)

Deploy the OpenAIRE-AI-Research-Evaluator with Helm on k3s.

## Prerequisites

- k3s cluster running
- `helm` installed
- Access to the Docker image `ethicalabs/openaire-research-evaluator:latest`
- [Request access](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF) to the gated Echo-DSRN model

## 1. Generate PostgreSQL dump

The SQLite golden database must be migrated to PostgreSQL:

```bash
./scripts/sqlite_to_postgres.sh
```

Generates `data/collaborative_pg.sql` — ready for PostgreSQL import.

## 2. Install

```bash
helm install evaluator ./helm/openaire-research-evaluator \
  --set secrets.hfToken=hf_... \
  --set secrets.secretKey=$(openssl rand -hex 32) \
  --set ingress.host=openaire-eval.local
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
```

```bash
helm install evaluator ./helm/openaire-research-evaluator -f values.local.yaml
```

## 3. Load golden data

Wait for PostgreSQL to be ready, then import:

```bash
kubectl cp data/collaborative_pg.sql \
  $(kubectl get pod -l component=postgres -o jsonpath='{.items[0].metadata.name}'):/tmp/collaborative_pg.sql

kubectl exec -it deployment/evaluator-openaire-research-evaluator-postgres -- \
  psql -U postgres -d echo_dsrn -f /tmp/collaborative_pg.sql
```

## 4. Access

```bash
kubectl port-forward svc/evaluator-openaire-research-evaluator-web 7860:7860
```

App at `http://localhost:7860`. With Ingress configured, at `http://openaire-eval.local`.

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
