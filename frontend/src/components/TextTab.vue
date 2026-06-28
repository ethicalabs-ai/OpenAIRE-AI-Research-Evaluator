<template>
  <div class="panel text-panel">
    <div class="panels-grid">
      <!-- Input Panel -->
      <div class="card panel-card input-card">
        <div class="card-header">
          <h3>Custom Paper Details</h3>
          <span class="card-badge">Input</span>
        </div>
        <div class="form-body">
          <div class="form-group">
            <label for="title-input">Paper Title</label>
            <input
              id="title-input"
              v-model="title"
              placeholder="e.g. Attention Is All You Need"
              class="cyber-input"
              :disabled="isLoading"
            />
          </div>
          <div class="form-group">
            <label for="abstract-input">Paper Abstract</label>
            <textarea
              id="abstract-input"
              v-model="abstract"
              placeholder="Type or paste the full abstract here (at least 50 characters)..."
              class="cyber-textarea"
              :disabled="isLoading"
            ></textarea>
          </div>
        </div>
        <div class="card-footer">
          <span class="char-count">{{ abstract.length }} chars</span>
          <div class="actions">
            <button
              @click="clearForm"
              class="btn btn-secondary"
              :disabled="isLoading || (!title && !abstract)"
            >
              Clear
            </button>
            <button
              @click="classify"
              class="btn btn-primary"
              :disabled="
                isLoading || !title.trim() || abstract.trim().length < 50
              "
            >
              <span v-if="isLoading" class="small-spinner"></span>
              <span v-else>Classify Intent</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Output Panel -->
      <div class="card panel-card output-card">
        <div class="card-header">
          <h3>Classification Result</h3>
          <span v-if="result" class="card-badge success">Complete</span>
          <span v-else class="card-badge">Awaiting</span>
        </div>
        <div class="output-container">
          <div v-if="result" class="result-body">
            <div class="predicted-header">
              <span class="lbl-desc">Predicted Primary Intent</span>
              <span class="primary-label">{{ result.label }}</span>
            </div>
            <div class="prob-list">
              <div
                v-for="(prob, name) in result.probabilities"
                :key="name"
                class="prob-row"
              >
                <div class="prob-labels">
                  <span class="prob-name">{{ name }}</span>
                  <span class="prob-val">{{ (prob * 100).toFixed(1) }}%</span>
                </div>
                <div class="prob-bar-track">
                  <div
                    class="prob-bar-fill"
                    :class="{ active: name === result.label }"
                    :style="{ width: `${prob * 100}%` }"
                  ></div>
                </div>
              </div>
            </div>
            <div class="save-box">
              <button @click="saveResult" class="btn btn-secondary save-btn">
                {{ isSaved ? "✓ Saved" : "Save to History" }}
              </button>
            </div>
          </div>
          <div v-else class="empty-output">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
              <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
            <p>Input a title and abstract and click Classify Intent.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from "vue";

export default {
  props: {
    isLoading: Boolean,
  },
  emits: ["classify", "save"],
  setup(props, { emit }) {
    const title = ref("");
    const abstract = ref("");
    const result = ref(null);
    const isSaved = ref(false);

    watch([title, abstract], () => {
      isSaved.value = false;
    });

    const clearForm = () => {
      title.value = "";
      abstract.value = "";
      result.value = null;
      isSaved.value = false;
    };

    const classify = async () => {
      emit("classify", {
        title: title.value,
        abstract: abstract.value,
        callback: (res) => {
          result.value = res;
          isSaved.value = false;
        },
      });
    };

    const saveResult = () => {
      if (!result.value || isSaved.value) return;
      emit("save", {
        title: title.value,
        abstract: abstract.value,
        label: result.value.label,
        probabilities: result.value.probabilities,
        timestamp: Date.now(),
      });
      isSaved.value = true;
    };

    return {
      title,
      abstract,
      result,
      isSaved,
      clearForm,
      classify,
      saveResult,
    };
  },
};
</script>

<style scoped>
.form-body {
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  flex: 1;
}
.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.form-group label {
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 500;
}
.cyber-input {
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 0.75rem 1rem;
  border-radius: 6px;
  font-family: inherit;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s ease;
}
.cyber-input:focus {
  border-color: var(--accent-blue);
}
.cyber-textarea {
  flex: 1;
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 0.75rem 1rem;
  border-radius: 6px;
  font-family: inherit;
  font-size: 0.95rem;
  min-height: 180px;
  resize: vertical;
  outline: none;
  transition: border-color 0.2s ease;
}
.cyber-textarea:focus {
  border-color: var(--accent-blue);
}
.result-body {
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  flex: 1;
}
.predicted-header {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 1rem;
}
.lbl-desc {
  font-size: 0.8rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.primary-label {
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent-blue);
  font-family: "Space Grotesk", sans-serif;
}
.prob-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
}
.prob-row {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.prob-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
}
.prob-name {
  color: var(--text-secondary);
  font-weight: 500;
}
.prob-val {
  color: var(--text-primary);
  font-family: monospace;
}
.prob-bar-track {
  height: 6px;
  background-color: var(--bg-tertiary);
  border-radius: 3px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 100%;
  background: var(--text-muted);
  border-radius: 3px;
  transition: width 0.5s ease-out;
}
.prob-bar-fill.active {
  background: var(--accent-glow);
}
.save-box {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  border-top: 1px solid var(--border-color);
  padding-top: 1.25rem;
  flex-wrap: wrap;
}
@media (max-width: 640px) {
  .save-box {
    justify-content: stretch;
  }
  .save-box button {
    flex: 1 1 100%;
  }
}
</style>
