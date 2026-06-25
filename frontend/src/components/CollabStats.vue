<template>
  <div class="glass-card stats-footer-card">
    <div class="stats-summary-grid">
      <div class="stat-box">
        <span class="stat-value">{{ stats.total_papers || 0 }}</span>
        <span class="stat-label">Papers Registered</span>
      </div>
      <div class="stat-box">
        <span class="stat-value">{{ stats.total_annotations || 0 }}</span>
        <span class="stat-label">Total Annotations</span>
      </div>
      <div class="stat-box">
        <span class="stat-value stat-human"
          >👤 {{ stats.human_annotations || 0 }}</span
        >
        <span class="stat-label">Human</span>
      </div>
      <div class="stat-box">
        <span class="stat-value stat-llm"
          >🤖 {{ stats.llm_annotations || 0 }}</span
        >
        <span class="stat-label">LLM Judge</span>
      </div>
      <div class="stat-box">
        <span class="stat-value">{{ stats.total_users || 0 }}</span>
        <span class="stat-label">Unique Curators</span>
      </div>
      <div class="stat-box">
        <span class="stat-value text-red">{{ stats.flagged_papers || 0 }}</span>
        <span class="stat-label">Papers Flagged</span>
      </div>
    </div>
    <div class="export-box border-left">
      <h4>💾 Gold-Standard dataset</h4>
      <p v-if="exportEnabled">
        Export the community-curated intent classification dataset in JSONL chat
        format.
      </p>
      <p v-else class="export-disabled-msg">{{ exportMessage }}</p>
      <button
        @click="$emit('export')"
        class="btn btn-accent btn-sm btn-block"
        :disabled="!exportEnabled"
      >
        Download JSONL
      </button>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from "vue";

export default {
  props: { stats: { type: Object, default: () => ({}) } },
  emits: ["export"],
  setup() {
    const exportEnabled = ref(false);
    const exportMessage = ref("");

    onMounted(async () => {
      try {
        const res = await fetch("/api/export/status");
        const data = await res.json();
        exportEnabled.value = data.enabled;
        exportMessage.value = data.message || "";
      } catch (e) {
        /* ignore */
      }
    });

    return { exportEnabled, exportMessage };
  },
};
</script>

<style scoped>
.glass-card {
  background: rgba(30, 41, 59, 0.45);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 1.5rem;
}
.stats-footer-card {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 2rem;
  align-items: center;
}
.stats-summary-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
}
.stat-box {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.stat-value {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent-blue);
  font-family: "Space Grotesk", sans-serif;
}
.stat-label {
  font-size: 0.8rem;
  color: var(--text-muted);
}
.stat-human {
  color: #60a5fa;
}
.stat-llm {
  color: #c084fc;
}
.text-red {
  color: #ef4444;
}
.export-box h4 {
  color: var(--text-primary);
  font-size: 0.95rem;
  margin-bottom: 0.4rem;
}
.export-box p {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-bottom: 0.75rem;
}
.export-disabled-msg {
  color: var(--text-muted);
  font-style: italic;
}
.border-left {
  border-left: 1px solid rgba(255, 255, 255, 0.08);
  padding-left: 1.5rem;
}
.btn {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 500;
  border-radius: 6px;
  padding: 0.6rem 1.2rem;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: center;
}
.btn-sm {
  font-size: 0.8rem;
  padding: 0.4rem 0.8rem;
}
.btn-block {
  width: 100%;
}
.btn-accent {
  background: var(--accent-blue);
  color: #1e293b;
  font-weight: 600;
}
.btn-accent:hover {
  background: #00d2da;
  box-shadow: 0 0 12px rgba(0, 242, 254, 0.3);
}
.btn-accent:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.btn-accent:disabled:hover {
  background: var(--accent-blue);
  box-shadow: none;
}

@media (max-width: 1024px) {
  .stats-footer-card {
    grid-template-columns: 1fr;
  }
  .border-left {
    border-left: none;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-left: 0;
    padding-top: 1.5rem;
    width: 100%;
  }
}
</style>
