<template>
  <div class="panel panel-right glass-card">
    <div v-if="!selectedPaper" class="no-selection">
      <div class="placeholder-icon">🗳️</div>
      <h3>Select a Research Paper</h3>
      <p>
        Choose a paper from the catalog list to view user annotations, check
        surprise metrics, flag errors, and register your classification vote.
      </p>
    </div>
    <div v-else class="paper-details scrollable">
      <div class="detail-header">
        <div class="meta-row">
          <span class="source-badge" :class="selectedPaper.source">{{
            selectedPaper.source.toUpperCase()
          }}</span>
          <span class="doi-label">DOI: {{ selectedPaper.doi }}</span>
        </div>
        <h2>{{ selectedPaper.title }}</h2>
      </div>

      <div class="abstract-box">
        <h4>Abstract</h4>
        <p>{{ selectedPaper.abstract }}</p>
      </div>

      <!-- Live Classification -->
      <div class="live-classification-panel border-bottom">
        <div class="live-clf-header">
          <span class="section-label">🤖 Model Classification</span>
          <span v-if="isClassifying" class="clf-status classifying"
            ><span class="small-spinner"></span> Classifying…</span
          >
          <span v-else-if="livePrediction" class="clf-status done"
            >✓
            <strong :class="livePrediction.label">{{
              livePrediction.label
            }}</strong></span
          >
        </div>
        <div v-if="isClassifying" class="clf-loading-bars">
          <div v-for="n in 5" :key="n" class="prob-row-skeleton"></div>
        </div>
        <div v-else-if="livePrediction" class="prob-list">
          <div
            v-for="(prob, name) in livePrediction.probabilities"
            :key="name"
            class="prob-row"
          >
            <div class="prob-labels">
              <span
                class="prob-name"
                :class="{ 'prob-name--active': name === livePrediction.label }"
                >{{ name }}</span
              >
              <span class="prob-val">{{ (prob * 100).toFixed(1) }}%</span>
            </div>
            <div class="prob-bar-track">
              <div
                class="prob-bar-fill"
                :class="{ active: name === livePrediction.label }"
                :style="{ width: `${prob * 100}%` }"
              ></div>
            </div>
          </div>
        </div>
        <div v-else-if="!selectedPaper.abstract" class="clf-empty">
          No abstract available — cannot classify.
        </div>
        <div v-else-if="selectedPaper.initial_intent" class="clf-empty">
          Live classification unavailable — stored prediction:
          <strong :class="selectedPaper.initial_intent">{{
            selectedPaper.initial_intent
          }}</strong>
        </div>
        <div v-else class="clf-empty">
          Classification unavailable for this record.
        </div>
        <div v-if="!selectedPaper.is_external_pending" class="consensus-row">
          <span class="box-label">Community Consensus:</span>
          <span
            class="intent-value consensus-value"
            :class="
              selectedPaper.consensus_label || selectedPaper.initial_intent
            "
          >
            {{
              selectedPaper.consensus_label ||
              selectedPaper.initial_intent ||
              "Awaiting Votes"
            }}
          </span>
        </div>
      </div>

      <!-- Community Annotations -->
      <div
        v-if="!selectedPaper.is_external_pending"
        class="annotations-section border-top"
      >
        <h3>
          💬 Community Annotations ({{ selectedPaper.annotations.length }})
        </h3>
        <div class="annotations-list-grid">
          <div
            v-for="anno in selectedPaper.annotations"
            :key="anno.id"
            class="annotation-bubble"
            :class="{ 'annotation-bubble--llm': anno.annotator_type === 'llm' }"
          >
            <div class="bubble-header">
              <div class="anno-author">
                <img
                  v-if="anno.annotator_type !== 'llm'"
                  :src="
                    anno.user.avatar_url ||
                    'https://api.dicebear.com/7.x/bottts/svg?seed=' +
                      anno.user.name
                  "
                  class="avatar-sm"
                />
                <span v-else class="avatar-sm avatar-llm">🤖</span>
                <div class="author-details">
                  <span class="author-name">
                    {{ anno.user.name }}
                    <span
                      v-if="anno.annotator_type === 'llm'"
                      class="annotator-tag annotator-tag--llm"
                      >LLM</span
                    >
                    <span v-else class="annotator-tag annotator-tag--human"
                      >Human</span
                    >
                  </span>
                  <span class="author-inst">{{ anno.user.institution }}</span>
                </div>
              </div>
              <div class="anno-label">
                <span
                  v-if="anno.proposed_label"
                  class="label-badge badge-sm"
                  :class="anno.proposed_label"
                  >{{ anno.proposed_label }}</span
                >
                <span v-if="anno.is_flagged" class="badge-flagged badge-sm"
                  >⚠️ Flagged: {{ anno.flag_reason || "Corrupted" }}</span
                >
              </div>
            </div>
            <p v-if="anno.comment" class="anno-comment">"{{ anno.comment }}"</p>
            <div class="bubble-footer">
              <span>{{ formatDate(anno.updated_at) }}</span>
            </div>
          </div>
          <div
            v-if="selectedPaper.annotations.length === 0"
            class="empty-annotations"
          >
            No custom annotations submitted yet. Be the first!
          </div>
        </div>
      </div>

      <!-- Voting Form -->
      <div class="vote-form-section border-top">
        <h3>
          🗳️
          {{
            selectedPaper.is_external_pending
              ? "📥 Import and Submit Your Annotation"
              : "Submit Your Annotation"
          }}
        </h3>
        <div v-if="!user.authenticated" class="vote-blocker">
          <p>
            Please log in with Hugging Face using the panel at the top to
            contribute your annotation.
          </p>
        </div>
        <div v-else-if="hubReadOnly" class="vote-blocker">
          <p v-if="archive">
            📦 This is a read-only archive of round
            <strong>{{ version }}</strong
            >. Annotation contributions for this round are closed.
          </p>
          <p v-else>
            📦 The collaborative hub is now archived. Download the curated
            datasets and the fine-tuned model on
            <a :href="hfDatasetsUrl" target="_blank" rel="noopener"
              >HuggingFace</a
            >.
          </p>
        </div>
        <form
          v-else
          @submit.prevent="
            $emit(
              selectedPaper.is_external_pending
                ? 'importAndAnnotate'
                : 'submitVote',
            )
          "
          class="vote-form"
        >
          <div class="form-group">
            <label>Proposed Intent Label</label>
            <div class="label-selectors">
              <button
                v-for="lbl in availableLabels"
                :key="lbl"
                type="button"
                class="btn btn-selector"
                :class="{
                  active: voteForm.proposed_label === lbl,
                  [lbl]: voteForm.proposed_label === lbl,
                }"
                @click="
                  $emit('update:voteForm', { ...voteForm, proposed_label: lbl })
                "
              >
                {{ lbl }}
              </button>
            </div>
          </div>
          <div class="form-group checkbox-group">
            <label class="checkbox-container">
              <input
                type="checkbox"
                :checked="voteForm.is_flagged"
                @change="
                  $emit('update:voteForm', {
                    ...voteForm,
                    is_flagged: $event.target.checked,
                  })
                "
              />
              <span class="checkmark"></span>
              Flag this paper record (corrupted text, wrong extraction, math
              heavy)
            </label>
          </div>
          <div v-if="voteForm.is_flagged" class="form-group fade-in">
            <label>Reason for Flagging</label>
            <input
              :value="voteForm.flag_reason"
              @input="
                $emit('update:voteForm', {
                  ...voteForm,
                  flag_reason: $event.target.value,
                })
              "
              type="text"
              placeholder="e.g. Garbled text, PDF decoding failure"
              class="cyber-input"
            />
          </div>
          <div class="form-group">
            <label>Comment / Critique (Optional)</label>
            <textarea
              :value="voteForm.comment"
              @input="
                $emit('update:voteForm', {
                  ...voteForm,
                  comment: $event.target.value,
                })
              "
              placeholder="Explain why you chose this classification..."
              rows="3"
              class="cyber-input"
            ></textarea>
          </div>
          <button type="submit" class="btn btn-accent">
            {{
              selectedPaper.is_external_pending
                ? "📥 Import & Submit Annotation"
                : "Submit Annotation"
            }}
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    selectedPaper: { type: Object, default: null },
    user: { type: Object, required: true },
    isClassifying: { type: Boolean, default: false },
    livePrediction: { type: Object, default: null },
    voteForm: { type: Object, required: true },
    availableLabels: {
      type: Array,
      default: () => [
        "Methodology",
        "Dataset",
        "Review",
        "Applied",
        "Theoretical",
        "Unclassifiable",
      ],
    },
    hubReadOnly: { type: Boolean, default: false },
    archive: { type: Boolean, default: false },
    version: { type: String, default: "" },
  },
  emits: ["submitVote", "importAndAnnotate", "update:voteForm"],
  computed: {
    hfDatasetsUrl() {
      return import.meta.env.VITE_HF_DATASETS_URL || "";
    },
  },
  methods: {
    formatDate(dateStr) {
      if (!dateStr) return "";
      const date = new Date(dateStr);
      return date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    },
  },
};
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}
.glass-card {
  background: rgba(30, 41, 59, 0.45);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 1.5rem;
}
.no-selection {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: var(--text-muted);
}
.placeholder-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: 0.3;
}
.no-selection p {
  max-width: 400px;
  font-size: 0.9rem;
}
.paper-details {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.detail-header h2 {
  color: var(--text-primary);
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.5rem;
  line-height: 1.3;
  margin-top: 0.5rem;
}
.meta-row {
  display: flex;
  gap: 1rem;
  align-items: center;
}
.source-badge {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  letter-spacing: 0.02em;
}
.source-badge.arxiv {
  background: rgba(179, 27, 27, 0.15);
  color: #fca5a5;
  border: 1px solid rgba(179, 27, 27, 0.3);
}
.source-badge.openaire {
  background: rgba(3, 105, 161, 0.15);
  color: #bae6fd;
  border: 1px solid rgba(3, 105, 161, 0.3);
}
.source-badge.custom {
  background: rgba(109, 40, 217, 0.15);
  color: #ddd6fe;
  border: 1px solid rgba(109, 40, 217, 0.3);
}
.doi-label {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-family: monospace;
}
.abstract-box {
  background: rgba(15, 23, 42, 0.25);
  border: 1px solid rgba(255, 255, 255, 0.04);
  border-radius: 8px;
  padding: 1rem;
}
.abstract-box h4 {
  color: var(--accent-blue);
  margin-bottom: 0.5rem;
}
.abstract-box p {
  color: var(--text-secondary);
  font-size: 0.95rem;
  line-height: 1.5;
}

/* Live Classification */
.live-classification-panel {
  padding: 1rem 0 1rem;
  margin-bottom: 0.75rem;
}
.border-bottom {
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.border-top {
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding-top: 1.25rem;
}
.live-clf-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}
.section-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.clf-status {
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.35rem;
}
.clf-status.classifying {
  color: var(--text-muted);
}
.clf-status.done {
  color: var(--accent-blue);
}
.prob-list {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.prob-row {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.prob-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
}
.prob-name {
  color: var(--text-secondary);
}
.prob-name--active {
  color: var(--text-primary);
  font-weight: 600;
}
.prob-val {
  font-family: monospace;
  color: var(--text-muted);
}
.prob-bar-track {
  height: 5px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 3px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 100%;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
  transition: width 0.5s ease-out;
}
.prob-bar-fill.active {
  background: var(--accent-glow);
}
.clf-loading-bars {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}
.prob-row-skeleton {
  height: 28px;
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.04) 25%,
    rgba(255, 255, 255, 0.08) 50%,
    rgba(255, 255, 255, 0.04) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.2s infinite;
  border-radius: 4px;
}
@keyframes shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
.clf-empty {
  font-size: 0.85rem;
  color: var(--text-muted);
  font-style: italic;
}
.consensus-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.75rem;
  padding-top: 0.6rem;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.box-label {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 0.4rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.consensus-value {
  font-size: 0.9rem;
  font-weight: 600;
}
.intent-value.Methodology {
  color: #4ade80;
}
.intent-value.Dataset {
  color: #c084fc;
}
.intent-value.Review {
  color: #facc15;
}
.intent-value.Applied {
  color: #60a5fa;
}
.intent-value.Theoretical {
  color: #fb7185;
}
.intent-value.Unclassifiable {
  color: #94a3b8;
}

/* Annotations */
.annotations-section h3 {
  color: var(--text-primary);
  font-size: 1.1rem;
  margin-bottom: 1rem;
}
.annotations-list-grid {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.annotation-bubble {
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.04);
  border-radius: 8px;
  padding: 0.85rem 1rem;
}
.annotation-bubble--llm {
  background: rgba(139, 92, 246, 0.06);
  border-color: rgba(139, 92, 246, 0.2);
}
.avatar-sm {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid var(--accent-blue);
  background: rgba(255, 255, 255, 0.1);
}
.avatar-llm {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  background: rgba(139, 92, 246, 0.15);
  border-color: rgba(139, 92, 246, 0.4);
  flex-shrink: 0;
}
.annotator-tag {
  display: inline-block;
  font-size: 0.6rem;
  font-weight: 700;
  padding: 0.05rem 0.35rem;
  border-radius: 4px;
  vertical-align: middle;
  margin-left: 0.3rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.annotator-tag--human {
  background: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
  border: 1px solid rgba(59, 130, 246, 0.3);
}
.annotator-tag--llm {
  background: rgba(139, 92, 246, 0.2);
  color: #c084fc;
  border: 1px solid rgba(139, 92, 246, 0.4);
}
.bubble-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}
.anno-author {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.author-details {
  display: flex;
  flex-direction: column;
}
.author-name {
  color: var(--text-primary);
  font-size: 0.85rem;
  font-weight: 600;
}
.author-inst {
  color: var(--text-muted);
  font-size: 0.75rem;
}
.anno-comment {
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.4;
  margin: 0.5rem 0 0.5rem 0;
  font-style: italic;
}
.bubble-footer {
  font-size: 0.75rem;
  color: var(--text-muted);
  display: flex;
  justify-content: flex-end;
}
.badge-sm {
  font-size: 0.65rem;
  padding: 0.1rem 0.35rem;
}
.badge-flagged {
  font-size: 0.65rem;
  font-weight: 600;
  background: rgba(239, 68, 68, 0.15);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
}
.label-badge {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 12px;
}
.label-badge.Methodology {
  background: rgba(34, 197, 94, 0.15);
  color: #86efac;
  border: 1px solid rgba(34, 197, 94, 0.3);
}
.label-badge.Dataset {
  background: rgba(147, 51, 234, 0.15);
  color: #d8b4fe;
  border: 1px solid rgba(147, 51, 234, 0.3);
}
.label-badge.Review {
  background: rgba(234, 179, 8, 0.15);
  color: #fef08a;
  border: 1px solid rgba(234, 179, 8, 0.3);
}
.label-badge.Applied {
  background: rgba(59, 130, 246, 0.15);
  color: #93c5fd;
  border: 1px solid rgba(59, 130, 246, 0.3);
}
.label-badge.Theoretical {
  background: rgba(244, 63, 94, 0.15);
  color: #fda4af;
  border: 1px solid rgba(244, 63, 94, 0.3);
}
.label-badge.Unclassifiable {
  background: rgba(148, 163, 184, 0.15);
  color: #cbd5e1;
  border: 1px solid rgba(148, 163, 184, 0.3);
}
.empty-annotations {
  color: var(--text-muted);
  font-style: italic;
  font-size: 0.85rem;
  text-align: center;
  padding: 1rem;
}

/* Vote Form */
.vote-form-section h3 {
  color: var(--text-primary);
  font-size: 1.1rem;
  margin-bottom: 1rem;
}
.vote-blocker {
  background: rgba(15, 23, 42, 0.3);
  border: 1px dashed rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.9rem;
}
.vote-form {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.form-group label {
  color: var(--text-secondary);
  font-size: 0.85rem;
  font-weight: 600;
}
.label-selectors {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.btn-selector {
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-secondary);
  font-size: 0.85rem;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.btn-selector:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-primary);
}
.btn-selector.active.Methodology {
  background: rgba(34, 197, 94, 0.2);
  border-color: #4ade80;
  color: #4ade80;
}
.btn-selector.active.Dataset {
  background: rgba(147, 51, 234, 0.2);
  border-color: #c084fc;
  color: #c084fc;
}
.btn-selector.active.Review {
  background: rgba(234, 179, 8, 0.2);
  border-color: #facc15;
  color: #facc15;
}
.btn-selector.active.Applied {
  background: rgba(59, 130, 246, 0.2);
  border-color: #60a5fa;
  color: #60a5fa;
}
.btn-selector.active.Theoretical {
  background: rgba(244, 63, 94, 0.2);
  border-color: #fb7185;
  color: #fb7185;
}
.btn-selector.active.Unclassifiable {
  background: rgba(148, 163, 184, 0.2);
  border-color: #94a3b8;
  color: #cbd5e1;
}
.checkbox-container {
  display: flex;
  align-items: center;
  position: relative;
  padding-left: 28px;
  cursor: pointer;
  font-size: 0.85rem !important;
  user-select: none;
}
.checkbox-container input {
  position: absolute;
  opacity: 0;
  cursor: pointer;
  height: 0;
  width: 0;
}
.checkmark {
  position: absolute;
  top: 0;
  left: 0;
  height: 18px;
  width: 18px;
  background-color: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}
.checkbox-container:hover input ~ .checkmark {
  border-color: rgba(255, 255, 255, 0.25);
}
.checkbox-container input:checked ~ .checkmark {
  background-color: var(--accent-blue);
  border-color: var(--accent-blue);
}
.checkmark:after {
  content: "";
  position: absolute;
  display: none;
}
.checkbox-container input:checked ~ .checkmark:after {
  display: block;
}
.checkbox-container .checkmark:after {
  left: 6px;
  top: 2px;
  width: 4px;
  height: 9px;
  border: solid #1e293b;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}
.fade-in {
  animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(3px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.scrollable {
  flex: 1;
  overflow-y: auto;
  padding-right: 0.25rem;
}
.scrollable::-webkit-scrollbar {
  width: 6px;
}
.scrollable::-webkit-scrollbar-track {
  background: transparent;
}
.scrollable::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}
.scrollable::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.25);
}

/* Shared UI */
.cyber-input {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-primary);
  border-radius: 6px;
  padding: 0.6rem 0.9rem;
  font-family: inherit;
  font-size: 0.9rem;
  outline: none;
  transition: all 0.2s ease;
}
.cyber-input:focus {
  border-color: var(--accent-blue);
  box-shadow: 0 0 8px rgba(0, 242, 254, 0.15);
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
.btn-accent {
  background: var(--accent-blue);
  color: #1e293b;
  font-weight: 600;
}
.btn-accent:hover {
  background: #00d2da;
  box-shadow: 0 0 12px rgba(0, 242, 254, 0.3);
}

@media (max-width: 640px) {
  .meta-row {
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .doi-label {
    word-break: break-all;
  }
}
</style>
