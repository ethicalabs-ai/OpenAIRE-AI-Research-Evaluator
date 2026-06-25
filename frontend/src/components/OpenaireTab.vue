<template>
  <div class="panel openaire-panel">
    <div class="panels-grid">
      <!-- OpenAIRE Input Card -->
      <div class="card panel-card input-card">
        <div class="card-header">
          <h3>OpenAIRE Graph Stream</h3>
          <span v-if="category" class="card-badge">{{ category }}</span>
          <span v-else class="card-badge">Standby</span>
        </div>
        <div class="openaire-body">
          <div v-if="title" class="paper-display">
            <h4 class="paper-title">{{ title }}</h4>
            <div class="metadata-row">
              <span v-if="link" class="paper-link">
                Source: <a :href="link" target="_blank">Publication Link</a>
              </span>
              <span v-if="lang" class="lang-tag">
                Lang: {{ lang.toUpperCase() }}
              </span>
              <span v-if="authors && authors.length" class="authors">
                by {{ authors.slice(0, 3).join(', ') }}{{ authors.length > 3 ? ' et al.' : '' }}
              </span>
            </div>
            <div class="abstract-box">
              <h5>Abstract</h5>
              <p>{{ abstract }}</p>
            </div>
          </div>
          <div v-else class="empty-openaire">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
              <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
            <h3>No Paper Loaded</h3>
            <p>Click below to pull a random computer science or machine learning publication from the OpenAIRE Graph API.</p>
          </div>
        </div>
        <div class="card-footer">
          <button
            @click="fetchRandomPaper"
            class="btn btn-secondary"
            :disabled="isLoading || isFetching"
          >
            <span v-if="isFetching" class="small-spinner"></span>
            <span v-else>Fetch OpenAIRE Paper</span>
          </button>
          <button
            v-if="title"
            @click="classify"
            class="btn btn-primary"
            :disabled="isLoading || isFetching"
          >
            <span v-if="isLoading" class="small-spinner"></span>
            <span v-else>Classify Intent</span>
          </button>
        </div>
      </div>

      <!-- Output Card -->
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
              <div v-for="(prob, name) in result.probabilities" :key="name" class="prob-row">
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
                {{ isSaved ? '✓ Saved' : 'Save to History' }}
              </button>
              <button @click="contributeToCollab" class="btn btn-accent save-btn" :disabled="!result">
                🗳️ Contribute to Collab Hub
              </button>
            </div>
          </div>
          <div v-else class="empty-output">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
              <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
            <p>Fetch a paper from OpenAIRE and click Classify Intent.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  props: {
    isLoading: Boolean,
  },
  emits: ['classify', 'save', 'error'],
  setup(props, { emit }) {
    const title = ref('')
    const abstract = ref('')
    const authors = ref([])
    const category = ref('')
    const link = ref('')
    const lang = ref('')
    const result = ref(null)
    const isFetching = ref(false)
    const isSaved = ref(false)

    const fetchRandomPaper = async () => {
      isFetching.value = true
      result.value = null
      isSaved.value = false
      title.value = ''
      abstract.value = ''
      authors.value = []
      category.value = ''
      link.value = ''
      lang.value = ''
      try {
        const res = await fetch('/api/openaire/random')
        if (!res.ok) throw new Error(`HTTP error ${res.status}`)
        const data = await res.json()
        title.value = data.title
        abstract.value = data.abstract
        authors.value = data.authors
        category.value = data.category
        link.value = data.link
        lang.value = data.lang
      } catch (err) {
        emit('error', `Failed to fetch from OpenAIRE: ${err.message}`)
      } finally {
        isFetching.value = false
      }
    }

    const classify = async () => {
      if (!title.value || !abstract.value) return
      emit('classify', {
        title: title.value,
        abstract: abstract.value,
        callback: (res) => {
          result.value = res
          isSaved.value = false
        }
      })
    }

    const saveResult = () => {
      if (!result.value || isSaved.value) return
      emit('save', {
        title: title.value,
        abstract: abstract.value,
        label: result.value.label,
        probabilities: result.value.probabilities,
        link: link.value,
        lang: lang.value,
        openaire: true,
        timestamp: Date.now(),
      })
      isSaved.value = true
    }

    const contributeToCollab = async () => {
      if (!result.value) return
      const paper = {
        title: title.value,
        abstract: abstract.value,
        label: result.value.label,
        doi: link.value ? link.value.split('/').pop() || '' : '',
        source: 'openaire',
      }
      localStorage.setItem('echo_pending_collab', JSON.stringify(paper))

      const meRes = await fetch('/api/auth/me').catch(() => null)
      const meData = meRes ? await meRes.json() : {}
      if (meData.authenticated) {
        window.location.href = '/#collab'
      } else {
        const loginRes = await fetch('/api/auth/login/hf')
        const loginData = await loginRes.json()
        if (loginData.auth_url) window.location.href = loginData.auth_url
      }
    }

    return {
      title,
      abstract,
      authors,
      category,
      link,
      lang,
      result,
      isFetching,
      isSaved,
      fetchRandomPaper,
      classify,
      saveResult,
      contributeToCollab,
    }
  }
}
</script>

<style scoped>
.openaire-body {
  padding: 1.5rem;
  flex: 1;
  display: flex;
  flex-direction: column;
}
.empty-openaire {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  gap: 1rem;
  padding: 3rem;
  text-align: center;
}
.empty-openaire h3 {
  color: var(--text-secondary);
}
.paper-display {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
}
.paper-title {
  font-size: 1.15rem;
  color: var(--text-primary);
  line-height: 1.4;
}
.metadata-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.75rem;
}
.lang-tag {
  color: var(--accent-purple);
  font-weight: 600;
}
.abstract-box {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.abstract-box h5 {
  font-size: 0.9rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.abstract-box p {
  font-size: 0.95rem;
  color: var(--text-secondary);
  line-height: 1.6;
  max-height: 200px;
  overflow-y: auto;
  padding-right: 0.5rem;
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
  font-family: 'Space Grotesk', sans-serif;
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
  border-top: 1px solid var(--border-color);
  padding-top: 1.25rem;
}
</style>
