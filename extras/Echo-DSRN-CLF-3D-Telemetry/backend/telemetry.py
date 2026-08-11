"""Per-token DSRN telemetry capture for the Echo-DSRN research-intent classifier.

Adapted from the original ``backend/server.py::/api/trace`` which streamed
telemetry from the causal model + PEFT adapter.  Here the same trace is driven
from ``EchoForSequenceClassification``: each step runs the shared ``EchoModel``
on the next token while carrying the recurrent state and attention KV cache
forward, extracting the per-layer slow state (``c_t``), surprise gates
(``lambda``), fast state (``h_t``) bins and synthetic attention links — plus
classification entropy from the classifier head over the pooled hidden state,
so the entropy core reflects prediction uncertainty as tokens accumulate.

There is deliberately no generation phase: the classifier cannot (and should
not) generate text, so every trace step is an input token (``is_generated``
is always ``False``).
"""

import logging
import os

import torch

log = logging.getLogger(__name__)

# Cap the rendered prefix length (incremental loop, so cost is O(T)).
MAX_TRACE_TOKENS = int(os.environ.get("ECHO_MAX_TRACE_TOKENS", "1024"))


def format_model_text(model, tokenizer, text: str) -> str:
    """Apply the model's baked-in chat templates exactly like ``classify()`` does."""
    sys_prompt = getattr(model.config, "system_prompt", None)
    usr_template = getattr(model.config, "user_template", None)
    if sys_prompt and usr_template:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_template.format(text=text)},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    return text


def shannon_entropy(logits) -> float:
    """H(p) = -sum p(x) log p(x) over the softmax of ``logits`` (float32-safe)."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return float(entropy.item())


def classifier_logits(model, hidden_states, attention_mask=None):
    """Run the classification head over the last-pooled hidden states.

    Mirrors the pooling in ``EchoForSequenceClassification.forward`` so the
    per-step entropy matches the model's actual prediction pathway.
    """
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
    else:
        seq_lengths = torch.full(
            (hidden_states.size(0),),
            hidden_states.size(1) - 1,
            dtype=torch.long,
            device=hidden_states.device,
        )
    pooled = hidden_states[
        torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lengths
    ]
    pooled = model.dropout(pooled)
    return model.classifier(pooled)


def fast_state_bins(h_state, num_bins: int = 16) -> list:
    """Compress the GRU fast state (``h_t``) into ``num_bins`` mean magnitudes."""
    if h_state is None:
        return [0.0] * num_bins
    if h_state.dim() > 2:
        h_state = h_state[0, -1].float().abs()
    else:
        h_state = h_state[0].float().abs()
    bin_size = max(1, len(h_state) // num_bins)
    return [float(h_state[b * bin_size : (b + 1) * bin_size].mean()) for b in range(num_bins)]


def attention_links(lambda_val: float) -> list:
    """Synthetic attention links proportional to surprise (as in the original app)."""
    links = []
    if lambda_val > 0.05:
        num_links = min(3, max(1, int(lambda_val * 5)))
        for _ in range(num_links):
            links.append({"idx": 0, "val": float(lambda_val)})
    return links


def probabilities_from_logits(model, logits) -> dict:
    """Softmax probabilities keyed by ``config.id2label``."""
    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)
    id2label = getattr(model.config, "id2label", {})
    return {
        id2label.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs.tolist())
    }


def compute_trace(model, tokenizer, text: str, max_tokens: int = MAX_TRACE_TOKENS):
    """Build the per-token telemetry trace for ``text``.

    Returns ``(trace, prediction)`` where ``prediction`` is the classifier
    verdict on the (truncated) rendered prefix, so the trace and the headline
    label always refer to the same input.
    """
    formatted = format_model_text(model, tokenizer, text)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs.input_ids[0][:max_tokens]

    num_layers = len(model.model.blocks)
    trace = []
    past_states = None
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(len(input_ids)):
            # Incremental step: feed one token at a time and carry the
            # recurrent state + attention KV cache forward. This is O(T)
            # instead of re-running the full prefix each step (O(T^2)),
            # so long abstracts can be traced without grinding to a halt.
            curr_input = input_ids[i].view(1, 1).to(device)
            model_out = model.model(
                curr_input,
                past_key_values=past_states,
                output_dsrn_telemetry=True,
            )

            # The cache wrapper is not list-indexable — unwrap to a plain list.
            pkv = model_out.past_key_values
            past_states = pkv.states if hasattr(pkv, "states") else pkv

            logits = classifier_logits(model, model_out.last_hidden_state)
            entropy = shannon_entropy(logits)
            confidence = float(torch.softmax(logits.float(), dim=-1).max().item())

            layers_data = []
            for layer_idx in range(num_layers):
                c_state = model_out.all_c_states[layer_idx][0].float().numpy().tolist()
                lambda_val = float(model_out.all_gate_stats[layer_idx][0, -1].item())
                h_state = past_states[layer_idx][0]
                layers_data.append(
                    {
                        "c_state": c_state,
                        "lambda": lambda_val,
                        "fast_state": fast_state_bins(h_state),
                        "attention": attention_links(lambda_val),
                    }
                )

            trace.append(
                {
                    "token": tokenizer.decode([input_ids[i]]),
                    "token_id": int(input_ids[i]),
                    "layers": layers_data,
                    "output_head": {
                        "attention": [{"idx": len(layers_data) - 1, "val": confidence}]
                    },
                    "is_generated": False,
                    "entropy": entropy,
                }
            )

        prediction = {
            "label": str(
                getattr(model.config, "id2label", {}).get(
                    int(logits.argmax(dim=-1).item()), "?"
                )
            ),
            "probabilities": probabilities_from_logits(model, logits),
        }

    return trace, prediction
