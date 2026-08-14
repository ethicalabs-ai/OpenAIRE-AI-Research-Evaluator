# Inference

> **Model:** [`ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF`](https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF) — public repo, 6-class (Methodology, Dataset, Review, Applied, Theoretical, Unclassifiable).

## Install

```bash
pip install git+https://github.com/ethicalabs-ai/Echo-DSRN.git
```

## Classify

```python
from echo_dsrn import EchoForSequenceClassification
from transformers import AutoTokenizer

model     = EchoForSequenceClassification.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF", trust_remote_code=True
)

label, probs = model.classify(
    "Title: Attention Is All You Need\n"
    "Abstract: We propose a new simple network architecture, the Transformer...",
    tokenizer=tokenizer,
)
print(label, probs)
# → Methodology  tensor([0.87, 0.03, 0.02, 0.06, 0.02, 0.00])
```

## Model Specs

| Property | Value |
|---|---|
| Architecture | Echo-DSRN (Recurrent Neural Network) |
| Parameters | 98,266,629 (~98M) |
| Layers | 8 DSRN blocks |
| Hidden dim | 512 |
| Attention heads | 4 |
| Vocab size | 32,017 tokens |
| Precision | bfloat16 |
| Classes | 6 (incl. Unclassifiable) |
| Inference | Fast on CPU |

> Run `curl /api/model/card` for live specs from the loaded model.
