<template>
  <div class="panel model-card-panel">
    <div v-if="isLoadingData" class="init-card">
      <div class="cyber-spinner"></div>
      <h3>Loading Model Specifications...</h3>
      <p>Fetching active parameter counts and runtime status from the intent classifier backend.</p>
    </div>

    <div v-else-if="errorMsg" class="init-card">
      <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </svg>
      <h3>Failed to Load Model Card</h3>
      <p>{{ errorMsg }}</p>
      <button @click="fetchModelCard" class="btn btn-secondary mt-4">Retry</button>
    </div>

    <div v-else class="panels-grid">
      <!-- Left side: Specs & Param Census -->
      <div class="left-col">
        <!-- 🏗️ Architecture Specs -->
        <div class="card panel-card spec-card mb-4">
          <div class="card-header">
            <h3>🏗️ Architecture Specifications</h3>
            <span class="card-badge success">{{ cardData.model_type.toUpperCase() }}-DSRN</span>
          </div>
          <div class="spec-body">
            <div class="spec-row">
              <span class="spec-label">Model Architecture</span>
              <span class="spec-value">Echo-DSRN (Recurrent Neural Network)</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Layers (DSRN Blocks)</span>
              <span class="spec-value">{{ cardData.num_layers }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Hidden Dimension</span>
              <span class="spec-value">{{ cardData.hidden_size }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Attention Heads</span>
              <span class="spec-value">{{ cardData.num_heads }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Vocabulary Size</span>
              <span class="spec-value">{{ cardData.vocab_size.toLocaleString() }} tokens</span>
            </div>
          </div>
        </div>

        <!-- 📊 Parameter Census -->
        <div class="card panel-card census-card">
          <div class="card-header">
            <h3>📊 Parameter Census</h3>
            <span class="card-badge">{{ formatNum(cardData.total_params) }} Total</span>
          </div>
          <div class="spec-body">
            <div class="spec-row highlight">
              <span class="spec-label">Total Parameters</span>
              <span class="spec-value">{{ cardData.total_params.toLocaleString() }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Embeddings</span>
              <span class="spec-value">{{ cardData.embedding_params.toLocaleString() }} ({{ (cardData.embedding_params/cardData.total_params*100).toFixed(1) }}%)</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">DSRN Blocks (RNN/MLP)</span>
              <span class="spec-value">{{ cardData.blocks_params.toLocaleString() }} ({{ (cardData.blocks_params/cardData.total_params*100).toFixed(1) }}%)</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Classifier Head</span>
              <span class="spec-value">{{ cardData.classifier_params.toLocaleString() }} (&lt;0.1%)</span>
            </div>
            <div class="spec-row border-top-glow">
              <span class="spec-label">Trainable Parameters</span>
              <span class="spec-value text-glow-blue">{{ cardData.trainable_params.toLocaleString() }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Frozen Parameters</span>
              <span class="spec-value">{{ cardData.frozen_params.toLocaleString() }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right side: Runtime status & Loaded consumption -->
      <div class="right-col">
        <!-- ⚡ Active Runtime Consumption -->
        <div class="card panel-card runtime-card mb-4">
          <div class="card-header">
            <h3>⚡ Active Runtime Consumption</h3>
            <span class="card-badge success">Online</span>
          </div>
          <div class="spec-body">
            <div class="spec-row">
              <span class="spec-label">Execution Device</span>
              <span class="spec-value device-tag">{{ cardData.device.toUpperCase() }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Tensor Precision</span>
              <span class="spec-value precision-tag">{{ cardData.dtype }}</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Estimated Memory Footprint</span>
              <span class="spec-value">{{ (cardData.total_params * (cardData.dtype === 'bfloat16' || cardData.dtype === 'float16' ? 2 : 4) / (1024 * 1024)).toFixed(1) }} MB</span>
            </div>
            <div class="spec-row">
              <span class="spec-label">Local Checkpoint Path</span>
              <span class="spec-value path-text" :title="cardData.model_path">{{ cardData.model_path }}</span>
            </div>
          </div>
        </div>

        <!-- 💻 Loaded Model Consumption -->
        <div class="card panel-card code-card">
          <div class="card-header">
            <h3>💻 Model Consumption Code</h3>
            <div class="header-right">
              <a href="https://huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF"
                 target="_blank" class="hf-link" title="Gated — request access">
                🤗 HF Repo
              </a>
              <span class="card-badge">Python</span>
            </div>
          </div>
          <div class="code-body">
            <pre><code><span class="comment"># Install the Echo-DSRN package</span>
pip install git+https://github.com/ethicalabs-ai/Echo-DSRN.git

<span class="comment"># Model is gated — request access at huggingface.co/ethicalabs/Echo-DSRN-v0.1.3-Research-Intent-CLF</span>

<span class="keyword">from</span> echo_dsrn <span class="keyword">import</span> EchoForSequenceClassification
<span class="keyword">from</span> transformers <span class="keyword">import</span> AutoTokenizer

<span class="comment"># 1. Load sequence classification model and tokenizer</span>
model_path = <span class="string">"{{ cardData.model_path }}"</span>
model = EchoForSequenceClassification.from_pretrained(
    model_path,
    trust_remote_code=<span class="boolean">True</span>,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=<span class="boolean">True</span>
)

<span class="comment"># 2. Run inference</span>
text = <span class="string">"Title: ... \nAbstract: ..."</span>
label, probs = model.classify(text, tokenizer=tokenizer)
print(<span class="string">f"Intent: {label}"</span>)</code></pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  setup() {
    const cardData = ref(null)
    const isLoadingData = ref(true)
    const errorMsg = ref('')

    const formatNum = (num) => {
      if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`
      if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`
      return num.toLocaleString()
    }

    const fetchModelCard = async () => {
      isLoadingData.value = true
      errorMsg.value = ''
      try {
        const res = await fetch('/api/model/card')
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}))
          throw new Error(detail.detail || `HTTP Error ${res.status}`)
        }
        cardData.value = await res.json()
      } catch (err) {
        errorMsg.value = err.message || 'Failed to connect to the backend specifications endpoint.'
      } finally {
        isLoadingData.value = false
      }
    }

    onMounted(() => {
      fetchModelCard()
    })

    return {
      cardData,
      isLoadingData,
      errorMsg,
      fetchModelCard,
      formatNum,
    }
  }
}
</script>

<style scoped>
.left-col, .right-col {
  display: flex;
  flex-direction: column;
}
.mb-4 {
  margin-bottom: 1.5rem;
}
.mt-4 {
  margin-top: 1rem;
}
.spec-body {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.spec-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.95rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  padding-bottom: 0.75rem;
}
.spec-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
.spec-row.highlight {
  font-weight: 600;
  color: var(--text-primary);
}
.spec-row.border-top-glow {
  border-top: 1px solid var(--border-color);
  padding-top: 1rem;
}
.spec-label {
  color: var(--text-secondary);
}
.spec-value {
  color: var(--text-primary);
  font-family: monospace;
}
.text-glow-blue {
  color: var(--accent-blue);
  text-shadow: 0 0 8px rgba(0, 242, 254, 0.3);
}
.device-tag {
  background-color: rgba(0, 242, 254, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(0, 242, 254, 0.2);
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-weight: 600;
}
.precision-tag {
  background-color: rgba(127, 0, 255, 0.1);
  color: var(--accent-purple);
  border: 1px solid rgba(127, 0, 255, 0.2);
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-weight: 600;
}
.path-text {
  max-width: 250px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.code-body {
  padding: 1.5rem;
  background-color: rgba(0, 0, 0, 0.2);
  font-family: 'Space Grotesk', monospace;
  font-size: 0.85rem;
  line-height: 1.5;
  overflow-x: auto;
  border-bottom-left-radius: 12px;
  border-bottom-right-radius: 12px;
}
pre {
  margin: 0;
}
.keyword { color: #f472b6; font-weight: 600; }
.string { color: #34d399; }
.comment { color: var(--text-muted); font-style: italic; }
.boolean { color: #fbbf24; }
.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.hf-link {
  color: var(--accent-blue);
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 500;
  transition: opacity 0.2s;
}
.hf-link:hover {
  opacity: 0.8;
}
</style>
