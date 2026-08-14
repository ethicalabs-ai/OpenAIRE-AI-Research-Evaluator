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
    <div v-if="stats.hub_read_only" class="readonly-banner">
      📦 The collaborative hub is now archived. Annotation contributions are
      closed. All curated datasets and models are available on HuggingFace.
    </div>
    <div class="round-banner">
      <template v-if="archiveVersion">
        📦 Viewing archive of round <strong>{{ archiveVersion }}</strong> ·
        active round: <strong>{{ stats.model_version || "…" }}</strong>
      </template>
      <template v-else>
        Active annotation round:
        <strong>{{ stats.model_version || "…" }}</strong>
      </template>
      <template v-if="stats.archive_versions && stats.archive_versions.length">
        · Archives:
        <a
          v-for="av in stats.archive_versions"
          :key="av"
          :href="`/collab/archive/${av}/`"
          class="archive-round-link"
          >{{ av }}</a
        >
      </template>
    </div>
    <div class="export-box border-left">
      <h4>📦 Curated Datasets</h4>
      <p>
        Download the LLM-as-Judge and gold-standard datasets, along with the
        current Echo-DSRN intent classifier
        (ethicalabs/Echo-DSRN-v0.1.4-Research-Intent-CLF).
      </p>
      <a
        :href="hfDatasetsUrl"
        target="_blank"
        rel="noopener"
        class="btn btn-accent btn-sm btn-block"
      >
        View on HuggingFace →
      </a>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    stats: { type: Object, default: () => ({}) },
    // Set when the collab tab is showing a read-only archive round.
    archiveVersion: { type: String, default: "" },
  },
  setup() {
    const hfDatasetsUrl = import.meta.env.VITE_HF_DATASETS_URL || "";
    return { hfDatasetsUrl };
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
.round-banner {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-bottom: 0.75rem;
}
.round-banner strong {
  color: var(--accent-blue);
}
.archive-round-link {
  margin-left: 0.35rem;
  color: var(--accent-blue);
  text-decoration: none;
  border-bottom: 1px dashed rgba(0, 242, 254, 0.4);
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
