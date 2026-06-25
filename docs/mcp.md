# MCP — Agent Classification Endpoint

Expose Echo-DSRN-98M research intent classification to AI agents via the Model Context Protocol (SSE transport).

## Server Configuration

Add to your MCP client config (Claude Desktop, Cursor, Continue, etc.):

```json
{
  "mcpServers": {
    "echo-dsrn": {
      "url": "http://localhost:7860/api/mcp/classify",
      "transport": "sse",
      "headers": {
        "Authorization": "Bearer echo-dsrn-mcp-change-me-in-production"
      }
    }
  }
}
```

Set `MCP_API_KEY` in `.env` to match:

```env
MCP_API_KEY=echo-dsrn-mcp-change-me-in-production
```

## Authentication

Bearer token via `Authorization` header. No query-string keys.

## Tools exposed

| Tool | Description |
|---|---|
| `classify_intent` | Classify a research paper by title + abstract into one of 5 intents |

### Input

```json
{
  "title": "Attention Is All You Need",
  "abstract": "We propose a new simple network architecture, the Transformer..."
}
```

### Output

```json
{
  "label": "Methodology",
  "probabilities": {
    "Methodology": 0.87,
    "Dataset": 0.03,
    "Review": 0.02,
    "Applied": 0.06,
    "Theoretical": 0.02
  }
}
```

## Raw SSE usage

```bash
curl -N -H "Authorization: Bearer echo-dsrn-mcp-change-me-in-production" \
  "http://localhost:7860/api/mcp/classify?title=Attention+Is+All+You+Need&abstract=We+propose+the+Transformer"
```

```python
import requests

resp = requests.get(
    "http://localhost:7860/api/mcp/classify",
    params={"title": "Attention Is All You Need", "abstract": "We propose..."},
    headers={"Authorization": "Bearer echo-dsrn-mcp-change-me-in-production"},
    stream=True,
)
for line in resp.iter_lines():
    if line:
        print(line.decode())
```

## Architecture

1. Request hits SSE endpoint → Bearer token validated
2. Celery task `classify_mcp` enqueued to Redis
3. Worker loads Echo-DSRN-98M (baked into Docker image, no HF download)
4. Sub-millisecond CPU inference → result streamed via SSE

No papers saved. No database writes. Stateless.
