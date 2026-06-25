<template>
  <div class="panel panel-left glass-card">
    <div class="panel-header">
      <h3>📚 Public Catalog</h3>
      <span class="badge badge-info">
        <template v-if="searchSource === 'local'">{{ filteredPapers.length }} / {{ papers.length }} Papers</template>
        <template v-else>{{ papers.length }} Papers</template>
      </span>
    </div>

    <div class="search-tabs">
      <button @click="$emit('update:searchSource', 'local')" class="tab-btn" :class="{ active: searchSource === 'local' }">Local</button>
      <button @click="$emit('update:searchSource', 'openaire')" class="tab-btn" :class="{ active: searchSource === 'openaire' }">OpenAIRE</button>
    </div>

    <div class="search-box-container">
      <input
        :value="searchQuery"
        @input="$emit('update:searchQuery', $event.target.value)"
        type="text"
        :placeholder="searchPlaceholder"
        class="cyber-input search-input"
        @keyup.enter="$emit('performSearch')"
      />
      <button
        v-if="searchSource !== 'local'"
        @click="$emit('performSearch')"
        class="btn btn-accent btn-sm search-action-btn"
        :disabled="isSearching"
      >
        <span v-if="isSearching" class="small-spinner"></span>
        <span v-else>Search</span>
      </button>
    </div>

    <!-- Filters (local mode only) -->
    <div v-if="searchSource === 'local'" class="filter-bar">
      <select :value="filterLabel" @change="$emit('update:filterLabel', $event.target.value)" class="filter-select">
        <option value="">All Labels</option>
        <option v-for="lbl in availableLabels" :key="lbl" :value="lbl">{{ lbl }}</option>
      </select>
      <select :value="filterSource" @change="$emit('update:filterSource', $event.target.value)" class="filter-select">
        <option value="">All Sources</option>
        <option value="arxiv">arXiv</option>
        <option value="openaire">OpenAIRE</option>
        <option value="custom">Custom</option>
      </select>
      <select :value="filterAnnotator" @change="$emit('update:filterAnnotator', $event.target.value)" class="filter-select">
        <option value="">All Annotators</option>
        <option value="human">👤 Human only</option>
        <option value="llm">🤖 LLM only</option>
      </select>
      <button
        class="filter-toggle"
        :class="{ active: filterFlagged }"
        @click="$emit('update:filterFlagged', !filterFlagged)"
      >⚠️ Flagged</button>
    </div>

    <div class="papers-list scrollable">
      <template v-if="searchSource === 'local'">
        <div
          v-for="paper in filteredPapers"
          :key="paper.doi"
          class="paper-card"
          :class="{ active: selectedPaper && selectedPaper.doi === paper.doi }"
          @click="$emit('selectPaper', paper.doi)"
        >
          <div class="paper-card-header">
            <span class="source-badge" :class="paper.source">{{ paper.source.toUpperCase() }}</span>
            <span class="doi-text">{{ paper.doi }}</span>
          </div>
          <h4 class="paper-card-title">{{ paper.title }}</h4>
          <div class="paper-card-footer">
            <span class="label-badge" :class="paper.consensus_label || paper.initial_intent">
              {{ paper.consensus_label || paper.initial_intent || 'Unclassified' }}
            </span>
            <div class="stats-indicators">
              <button class="save-btn" title="Save to history" @click.stop="$emit('saveToHistory', paper)">📋</button>
              <span class="indicator-item" title="Community votes">🗳️ {{ paper.vote_count }}</span>
              <span v-if="paper.flag_count > 0" class="indicator-item text-red" title="Reported issues">⚠️ {{ paper.flag_count }}</span>
            </div>
          </div>
        </div>
        <div v-if="filteredPapers.length === 0" class="empty-state">No papers matched your search.</div>
      </template>

      <template v-else>
        <div
          v-for="paper in externalResults"
          :key="paper.doi"
          class="paper-card external-card"
          :class="{ active: selectedPaper && selectedPaper.doi === paper.doi }"
          @click="$emit('selectExternalPaper', paper)"
        >
          <div class="paper-card-header">
            <span class="source-badge" :class="paper.source">{{ paper.source.toUpperCase() }}</span>
            <span class="doi-text">{{ paper.doi }}</span>
          </div>
          <h4 class="paper-card-title">{{ paper.title }}</h4>
          <div class="paper-card-footer">
            <span class="label-badge external-label">Stream Result</span>
            <div class="stats-indicators">
              <button class="save-btn" title="Save to history" @click.stop="$emit('saveToHistory', paper)">📋</button>
              <span class="action-hint">Click to Import &amp; Vote</span>
            </div>
          </div>
        </div>
        <div v-if="isSearching" class="empty-state"><span class="spinner"></span> Searching external repository...</div>
        <div v-else-if="externalResults.length === 0" class="empty-state">
          {{ searchQuery ? 'No external papers found.' : 'Enter a query and press Enter / click Search.' }}
        </div>
      </template>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    papers: { type: Array, default: () => [] },
    filteredPapers: { type: Array, default: () => [] },
    externalResults: { type: Array, default: () => [] },
    searchQuery: { type: String, default: '' },
    searchSource: { type: String, default: 'local' },
    searchPlaceholder: { type: String, default: '' },
    isSearching: { type: Boolean, default: false },
    selectedPaper: { type: Object, default: null },
    filterLabel: { type: String, default: '' },
    filterSource: { type: String, default: '' },
    filterFlagged: { type: Boolean, default: false },
    filterAnnotator: { type: String, default: '' },
    availableLabels: { type: Array, default: () => [] },
  },
  emits: ['update:searchQuery', 'update:searchSource', 'performSearch', 'selectPaper', 'selectExternalPaper', 'saveToHistory',
          'update:filterLabel', 'update:filterSource', 'update:filterFlagged', 'update:filterAnnotator'],
}
</script>

<style scoped>
.panel { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.panel-header h3 { color: var(--text-primary); font-size: 1.2rem; margin: 0; }
.glass-card {
  background: rgba(30, 41, 59, 0.45);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 1.5rem;
}
.search-tabs {
  display: flex;
  background: rgba(15, 23, 42, 0.45);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 2px;
  margin-bottom: 0.75rem;
}
.tab-btn {
  flex: 1; background: transparent; border: none; border-radius: 6px;
  color: var(--text-muted); font-size: 0.85rem; font-weight: 500;
  padding: 0.45rem 0; cursor: pointer; transition: all 0.2s ease;
}
.tab-btn:hover { color: var(--text-primary); background: rgba(255, 255, 255, 0.02); }
.tab-btn.active {
  background: rgba(0, 242, 254, 0.12); color: var(--accent-blue);
  border: 1px solid rgba(0, 242, 254, 0.2); box-shadow: 0 0 10px rgba(0, 242, 254, 0.15); font-weight: 600;
}
.search-box-container { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.search-input { flex: 1; width: 100%; }
.search-action-btn { white-space: nowrap; }

.filter-bar { display: flex; gap: 0.4rem; margin-bottom: 0.75rem; flex-wrap: wrap; }
.filter-select {
  flex: 1; min-width: 0;
  background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-secondary); border-radius: 6px; padding: 0.35rem 0.5rem;
  font-family: inherit; font-size: 0.78rem; outline: none; cursor: pointer;
  transition: all 0.2s ease;
}
.filter-select:focus { border-color: var(--accent-blue); color: var(--text-primary); }
.filter-toggle {
  background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-muted); border-radius: 6px; padding: 0.35rem 0.65rem;
  font-family: inherit; font-size: 0.78rem; cursor: pointer; white-space: nowrap;
  transition: all 0.2s ease;
}
.filter-toggle:hover { border-color: rgba(255, 255, 255, 0.2); color: var(--text-primary); }
.filter-toggle.active { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); color: #fca5a5; }

.papers-list { display: flex; flex-direction: column; gap: 0.75rem; }
.paper-card {
  background: rgba(15, 23, 42, 0.35); border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 8px; padding: 0.85rem 1rem; cursor: pointer;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
.paper-card:hover { background: rgba(15, 23, 42, 0.6); border-color: rgba(255, 255, 255, 0.12); transform: translateY(-1px); }
.paper-card.active { border-color: var(--accent-blue); background: rgba(0, 242, 254, 0.04); }
.paper-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.source-badge { font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.4rem; border-radius: 4px; letter-spacing: 0.02em; }
.source-badge.arxiv { background: rgba(179, 27, 27, 0.15); color: #fca5a5; border: 1px solid rgba(179, 27, 27, 0.3); }
.source-badge.openaire { background: rgba(3, 105, 161, 0.15); color: #bae6fd; border: 1px solid rgba(3, 105, 161, 0.3); }
.source-badge.custom { background: rgba(109, 40, 217, 0.15); color: #ddd6fe; border: 1px solid rgba(109, 40, 217, 0.3); }
.doi-text { font-family: monospace; font-size: 0.75rem; color: var(--text-muted); }
.paper-card-title {
  color: var(--text-primary); font-size: 0.95rem; line-height: 1.35; margin-bottom: 0.65rem;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.paper-card-footer { display: flex; justify-content: space-between; align-items: center; }
.label-badge { font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 12px; }
.label-badge.Methodology { background: rgba(34, 197, 94, 0.15); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.3); }
.label-badge.Dataset { background: rgba(147, 51, 234, 0.15); color: #d8b4fe; border: 1px solid rgba(147, 51, 234, 0.3); }
.label-badge.Review { background: rgba(234, 179, 8, 0.15); color: #fef08a; border: 1px solid rgba(234, 179, 8, 0.3); }
.label-badge.Applied { background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3); }
.label-badge.Theoretical { background: rgba(244, 63, 94, 0.15); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.3); }
.stats-indicators { display: flex; gap: 0.5rem; font-size: 0.8rem; color: var(--text-secondary); }
.text-red { color: #ef4444; }
.save-btn {
  background: none; border: 1px solid rgba(255, 255, 255, 0.12); color: var(--text-muted);
  border-radius: 4px; padding: 2px 6px; cursor: pointer; font-size: 0.85rem;
  transition: all 0.15s ease; line-height: 1;
}
.save-btn:hover { background: rgba(0, 242, 254, 0.12); border-color: var(--accent-blue); color: var(--accent-blue); }
.external-card { border-style: dashed !important; }
.external-label { background: rgba(168, 85, 247, 0.15) !important; color: var(--accent-purple) !important; border: 1px solid rgba(168, 85, 247, 0.25); }
.action-hint { font-size: 0.75rem; color: var(--accent-glow); font-weight: 500; }

.empty-state { color: var(--text-muted); text-align: center; padding: 2rem 0; }
.scrollable { flex: 1; overflow-y: auto; padding-right: 0.25rem; }
.scrollable::-webkit-scrollbar { width: 6px; }
.scrollable::-webkit-scrollbar-track { background: transparent; }
.scrollable::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }
.scrollable::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }

/* Shared UI */
.cyber-input {
  background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-primary); border-radius: 6px; padding: 0.6rem 0.9rem;
  font-family: inherit; font-size: 0.9rem; outline: none; transition: all 0.2s ease;
}
.cyber-input:focus { border-color: var(--accent-blue); box-shadow: 0 0 8px rgba(0, 242, 254, 0.15); }
.btn {
  font-family: 'Space Grotesk', sans-serif; font-weight: 500; border-radius: 6px;
  padding: 0.6rem 1.2rem; cursor: pointer; border: 1px solid transparent;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); text-align: center;
}
.btn-sm { font-size: 0.8rem; padding: 0.4rem 0.8rem; }
.btn-accent { background: var(--accent-blue); color: #1e293b; font-weight: 600; }
.btn-accent:hover { background: #00d2da; box-shadow: 0 0 12px rgba(0, 242, 254, 0.3); }
.badge { font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 12px; }
.badge-info { background: rgba(0, 242, 254, 0.08); color: var(--accent-blue); border: 1px solid rgba(0, 242, 254, 0.2); }
</style>
