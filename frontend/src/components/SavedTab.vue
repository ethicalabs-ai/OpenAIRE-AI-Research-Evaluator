<template>
  <div class="panel saved-panel">
    <div class="card full-width-card">
      <div class="card-header">
        <h3>Private Search History</h3>
        <span class="card-badge success">{{ items.length }} Saved</span>
      </div>
      <div class="saved-body">
        <div v-if="items.length === 0" class="empty-history">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <h3>History is Empty</h3>
          <p>
            Classified papers saved from the other tabs will be listed here. All
            data stays local to your browser.
          </p>
        </div>
        <div v-else class="history-list">
          <div
            v-for="(item, idx) in items"
            :key="item.timestamp"
            class="history-item"
          >
            <div class="item-summary" @click="toggleItem(idx)">
              <div class="item-left">
                <span class="badge" :class="item.label.toLowerCase()">{{
                  item.label
                }}</span>
                <span class="item-title">{{ item.title }}</span>
              </div>
              <div class="item-right">
                <span v-if="item.arxivId" class="item-tag">arXiv</span>
                <span v-else-if="item.openaire" class="item-tag openaire"
                  >OpenAIRE</span
                >
                <span v-else class="item-tag custom">Text</span>
                <span class="item-time">{{ formatTime(item.timestamp) }}</span>
                <button
                  @click.stop="contributeToCollab(idx)"
                  class="contribute-btn"
                  title="Contribute to Collab Hub"
                >
                  🗳️
                </button>
                <button
                  @click.stop="deleteItem(idx)"
                  class="delete-btn"
                  title="Delete"
                >
                  &times;
                </button>
              </div>
            </div>
            <div v-if="expanded[idx]" class="item-details">
              <p class="details-abstract">
                <strong>Abstract:</strong> {{ item.abstract }}
              </p>
              <div v-if="item.link" class="details-link">
                <strong>Link: </strong>
                <a :href="item.link" target="_blank">{{
                  item.arxivId || "Open Publication"
                }}</a>
              </div>
              <div class="details-probs">
                <strong>Softmax Distribution:</strong>
                <div class="probs-grid">
                  <div
                    v-for="(prob, name) in item.probabilities"
                    :key="name"
                    class="prob-chip"
                  >
                    <span class="chip-name">{{ name }}</span>
                    <span class="chip-val">{{ (prob * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from "vue";

export default {
  props: {
    items: {
      type: Array,
      required: true,
    },
  },
  emits: ["delete"],
  setup(props, { emit }) {
    const expanded = ref({});

    const toggleItem = (idx) => {
      expanded.value[idx] = !expanded.value[idx];
    };

    const deleteItem = (idx) => {
      emit("delete", { idx, item: props.items[idx] });
    };

    const contributeToCollab = async (idx) => {
      const item = props.items[idx];
      const paper = {
        title: item.title,
        abstract: item.abstract || "",
        label: item.label,
        doi: item.doi || item.link || "",
        source: item.openaire ? "openaire" : "custom",
      };
      localStorage.setItem("echo_pending_collab", JSON.stringify(paper));

      const meRes = await fetch("/api/auth/me").catch(() => null);
      const meData = meRes ? await meRes.json() : {};
      if (meData.authenticated) {
        window.history.pushState(null, "", "/collab");
        window.dispatchEvent(new PopStateEvent("popstate"));
      } else {
        const loginRes = await fetch("/api/auth/login/hf");
        const loginData = await loginRes.json();
        if (loginData.auth_url) window.location.href = loginData.auth_url;
      }
    };

    const formatTime = (ts) => {
      return new Date(ts).toLocaleString();
    };

    return {
      expanded,
      toggleItem,
      deleteItem,
      formatTime,
      contributeToCollab,
    };
  },
};
</script>

<style scoped>
.full-width-card {
  width: 100%;
}
.saved-body {
  padding: 1.5rem;
  min-height: 400px;
}
.empty-history {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  gap: 1rem;
  padding: 4rem;
  text-align: center;
}
.empty-history h3 {
  color: var(--text-secondary);
}
.history-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.history-item {
  border: 1px solid var(--border-color);
  background-color: var(--bg-tertiary);
  border-radius: 8px;
  overflow: hidden;
  transition: border-color 0.2s ease;
}
.history-item:hover {
  border-color: rgba(255, 255, 255, 0.15);
}
.item-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  cursor: pointer;
}
.item-left {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
  min-width: 0;
}
.badge {
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  text-transform: uppercase;
}
.badge.methodology {
  background-color: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
}
.badge.dataset {
  background-color: rgba(16, 185, 129, 0.15);
  color: #34d399;
}
.badge.review {
  background-color: rgba(139, 92, 246, 0.15);
  color: #a78bfa;
}
.badge.applied {
  background-color: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
}
.badge.theoretical {
  background-color: rgba(236, 72, 153, 0.15);
  color: #f472b6;
}

.item-title {
  color: var(--text-primary);
  font-size: 0.95rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.item-right {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-left: 1rem;
}
.item-tag {
  font-size: 0.7rem;
  color: var(--accent-blue);
  border: 1px solid rgba(0, 242, 254, 0.2);
  background-color: rgba(0, 242, 254, 0.05);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  text-transform: uppercase;
}
.item-tag.openaire {
  color: var(--accent-purple);
  border-color: rgba(127, 0, 255, 0.25);
  background-color: rgba(127, 0, 255, 0.05);
}
.item-tag.custom {
  color: var(--text-secondary);
  border-color: var(--border-color);
  background-color: rgba(255, 255, 255, 0.02);
}
.item-time {
  font-size: 0.8rem;
  color: var(--text-muted);
}
.delete-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 1.5rem;
  cursor: pointer;
  line-height: 1;
}
.delete-btn:hover {
  color: #ef4444;
}
.contribute-btn {
  background: transparent;
  border: 1px solid rgba(0, 242, 254, 0.15);
  color: var(--text-muted);
  font-size: 0.85rem;
  cursor: pointer;
  border-radius: 4px;
  padding: 2px 5px;
  transition: all 0.15s ease;
}
.contribute-btn:hover {
  border-color: var(--accent-blue);
  color: var(--accent-blue);
  background: rgba(0, 242, 254, 0.08);
}
.item-details {
  border-top: 1px solid var(--border-color);
  background-color: rgba(0, 0, 0, 0.15);
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.details-abstract {
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.6;
}
.details-link {
  font-size: 0.85rem;
  color: var(--text-secondary);
}
.details-probs {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}
.probs-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.prob-chip {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  display: flex;
  gap: 0.5rem;
}
.chip-name {
  color: var(--text-muted);
}
.chip-val {
  color: var(--text-primary);
  font-family: monospace;
}

@media (max-width: 640px) {
  .item-summary {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
  }
  .item-left {
    width: 100%;
  }
  .item-title {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
  }
  .item-right {
    margin-left: 0;
    width: 100%;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .item-time {
    font-size: 0.75rem;
  }
}
</style>
