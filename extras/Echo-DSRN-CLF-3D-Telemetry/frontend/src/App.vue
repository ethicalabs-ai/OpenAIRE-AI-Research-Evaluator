<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import TelemetryScene from './components/TelemetryScene.vue'

const trace = ref([])
const currentIndex = ref(0)
const loading = ref(true)
const loadingText = ref('INITIALIZING ECHO-DSRN NEURAL VOLUMES...')
const doiInput = ref('')
const paper = ref(null)
const prediction = ref(null)
const config = ref({ layers: 8, model_id: 'ECHO-DSRN-CLF' })
const footerText = import.meta.env.VITE_FOOTER_TEXT || 'OpenAIRE Graph API data: CC-BY 4.0.'
const logs = ref([])
const isPlaying = ref(false)
let playInterval = null
const playbackSpeed = ref(1.0) // 0.5x to 4x
let lastAction = null // most recent paper-fetch action, for RETRY

const selectedVoxel = ref(null)
const selectedGantry = ref(null)
const selectedEntropy = ref(null)
const selectedFilament = ref(null)

const mainCollapsed = ref(false)
const logCollapsed = ref(false)
const statsCollapsed = ref(false)

// Error handling state
const errorState = ref({
  hasError: false,
  message: '',
  retryCount: 0,
  maxRetries: 3,
  retryDelay: 1000
})

// Persistence keys
const STORAGE_KEYS = {
  DOI: 'echo_doi',
  TRACE: 'echo_trace',
  INDEX: 'echo_index',
  SPEED: 'echo_speed',
  PREFS: 'echo_prefs'
}

const gantryValue = computed(() => {
  if (!selectedGantry.value || !currentStep.value) return 0
  return currentStep.value.layers?.[selectedGantry.value.layerIndex]?.lambda || 0
})

const selectedFilamentValue = computed(() => {
  if (!selectedFilament.value || !currentStep.value) return 0
  const layerIdx = selectedFilament.value.layerIndex
  if (layerIdx === 'OUT' || layerIdx === 'OUTPUT') {
    return currentStep.value.output_head?.attention?.[0]?.val || 0
  } else {
    return currentStep.value.layers?.[layerIdx]?.attention?.[0]?.val || 0
  }
})

const selectedFilamentCount = computed(() => {
  if (!selectedFilament.value || !currentStep.value) return 0
  const layerIdx = selectedFilament.value.layerIndex
  if (layerIdx === 'OUT' || layerIdx === 'OUTPUT') {
    return currentStep.value.output_head?.attention?.length || 0
  } else {
    return currentStep.value.layers?.[layerIdx]?.attention?.length || 0
  }
})

const selectedFilamentSource = computed(() => {
  if (!selectedFilament.value || !currentStep.value) return 'UNKNOWN'
  const layerIdx = selectedFilament.value.layerIndex
  if (layerIdx === 'OUT' || layerIdx === 'OUTPUT') {
    const srcLayer = currentStep.value.output_head?.attention?.[0]?.idx
    return srcLayer !== undefined ? ('LAYER ' + srcLayer) : 'FINAL LAYER'
  }
  return 'INPUT CONTEXT'
})

const totalNetworkLinks = computed(() => {
  if (!currentStep.value) return 0
  let count = 0
  currentStep.value.layers?.forEach(L => { count += L.attention?.length || 0 })
  count += currentStep.value.output_head?.attention?.length || 0
  return count
})
const selectedVoxelValue = computed(() => {
  if (!selectedVoxel.value || !currentStep.value) return 0
  const layer = currentStep.value.layers?.[selectedVoxel.value.layerIndex]
  if (!layer || !layer.c_state) return 0
  const stateDim = layer.c_state.length
  if (stateDim === 0) return 0
  const idx = (selectedVoxel.value.voxelIndex * 4) % stateDim
  return layer.c_state[idx] || 0
})

const fastStateBins = computed(() => {
  if (!selectedVoxel.value || !currentStep.value) return Array(16).fill(0)
  return currentStep.value.layers?.[selectedVoxel.value.layerIndex]?.fast_state || Array(16).fill(0)
})

const fastStateVariance = computed(() => {
  const bins = fastStateBins.value
  if (!bins || bins.length === 0) return 0
  return bins.reduce((a, b) => a + b, 0) / bins.length
})

const cleanToken = (t) => {
  if (t === undefined || t === null) return ''
  // Strip quotes and brackets that might come from raw decoding/stringification
  return String(t).replace(/<\|.*?>/g, '').replace(/[\[\]"]/g, '').trim()
}

const predictedIntent = computed(() => {
  if (!prediction.value || !prediction.value.label) return 'READY'
  const label = prediction.value.label
  const conf = prediction.value.probabilities?.[label]
  return conf !== undefined
    ? `[${label.toUpperCase()}] ${(conf * 100).toFixed(1)}%`
    : `[${label.toUpperCase()}]`
})

const addLog = (msg) => {
  const timestamp = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  logs.value.unshift(`[${timestamp}] ${msg}`)
  if (logs.value.length > 20) logs.value.pop()
}

const currentStep = computed(() => trace.value[currentIndex.value] || null)

const resetError = () => {
  errorState.value = {
    hasError: false,
    message: '',
    retryCount: 0,
    maxRetries: 3,
    retryDelay: 1000
  }
}

const addErrorLog = (msg) => {
  addLog(msg)
  if (!errorState.value.hasError) {
    errorState.value.hasError = true
  }
}

const retryWithBackoff = async (fn, maxRetries = 8, initialDelay = 1500) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn()
    } catch (err) {
      if (i === maxRetries - 1) throw err
      const delay = initialDelay * Math.pow(1.5, i)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
}

const checkHealth = async () => {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error('Not healthy')
  const data = await res.json()
  if (data.status !== 'healthy') throw new Error('Not healthy')
  return true
}

const fetchConfig = async () => {
  try {
    // Check if backend is healthy, retrying gracefully while PEFT loads
    await retryWithBackoff(checkHealth, 8, 2000)
  } catch (err) {
    resetError()
    errorState.value.message = `Backend not responding. Ensure server.py is running on port 5005.`
    addErrorLog(`BACKEND_UNAVAILABLE: Cannot connect to /api/health`)
    loading.value = false
    return
  }

  try {
    await retryWithBackoff(async () => {
      const res = await fetch('/api/config')
      const data = await res.json()
      if (!data.error) config.value = data
      else throw new Error(data.error || 'Config fetch failed')
    })
  } catch (err) {
    resetError()
    errorState.value.message = `Backend connection failed: ${err.message}. Check if server is running on port 5005.`
    addErrorLog(`CONFIG_ERROR: ${err.message}`)
    loading.value = false
  }
}

const fetchRandomPaper = async () => {
  resetError()
  loadingText.value = 'FETCHING RANDOM PAPER FROM OPENAIRE...'
  loading.value = true
  lastAction = fetchRandomPaper
  try {
    const paperData = await retryWithBackoff(async () => {
      const res = await fetch('/api/openaire/random')
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
        throw new Error(errData.error || `Request failed: ${res.status}`)
      }
      return res.json()
    }, errorState.value.maxRetries)
    await fetchTrace(paperData)
  } catch (err) {
    resetError()
    errorState.value.message = `Paper fetch failed: ${err.message}.`
    addErrorLog(`PAPER_FETCH_ERROR: ${err.message}`)
  } finally {
    loading.value = false
  }
}

const fetchPaperByDoi = async (doi) => {
  const cleaned = (doi || '').trim()
  if (!cleaned) return
  resetError()
  loadingText.value = 'LOOKING UP DOI ON OPENAIRE...'
  loading.value = true
  lastAction = () => fetchPaperByDoi(cleaned)
  try {
    const paperData = await retryWithBackoff(async () => {
      const res = await fetch(`/api/openaire/search?doi=${encodeURIComponent(cleaned)}`)
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
        throw new Error(errData.error || `Request failed: ${res.status}`)
      }
      return res.json()
    }, errorState.value.maxRetries)
    await fetchTrace(paperData)
  } catch (err) {
    resetError()
    errorState.value.message = `DOI lookup failed: ${err.message}.`
    addErrorLog(`DOI_ERROR: ${err.message}`)
  } finally {
    loading.value = false
  }
}

const fetchTrace = async (paperData) => {
  resetError()
  if (!paperData?.title || !paperData?.abstract) {
    errorState.value.message = 'Paper has no abstract to classify.'
    addErrorLog('NO_ABSTRACT: paper missing abstract')
    return
  }

  loadingText.value = 'COMPUTING NEURAL TRACE...'
  loading.value = true
  lastAction = () => fetchTrace(paperData)
  try {
    await retryWithBackoff(async () => {
      const res = await fetch('/api/trace', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper: { title: paperData.title, abstract: paperData.abstract } })
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
        throw new Error(errData.error || `Request failed: ${res.status}`)
      }

      const data = await res.json()
      if (!data.trace || !Array.isArray(data.trace) || data.trace.length === 0) {
        throw new Error('Invalid trace data received')
      }
      trace.value = data.trace
      prediction.value = data.prediction || null
      paper.value = paperData
      currentIndex.value = 0
      addLog(`TRACE GENERATED: "${data.trace[0]?.token || ''}..." (${data.trace.length} tokens)`)
      saveState() // Save after successful trace generation
      // Auto-play the token trajectory so the 3D scene comes alive immediately
      setTimeout(() => {
        if (trace.value.length > 0) {
          firstStep()
          togglePlay(true)
        }
      }, 800)
    }, errorState.value.maxRetries)
  } catch (err) {
    resetError()
    errorState.value.message = `Trace generation failed: ${err.message}. Try another paper or check backend logs.`
    addErrorLog(`TRACE_ERROR: ${err.message}`)
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  // Load persisted state first
  loadState()
  
  // Then fetch config
  await fetchConfig()
  loading.value = false // Exit loading state once hardware config is locked
})

// Save state on significant changes
watch(() => doiInput.value, () => saveState())
watch(() => trace.value, () => saveState(), { deep: true })
watch(() => currentIndex.value, () => saveState())

// ── Draggable glass panels ───────────────────────────────────────────────────
// Panels are positioned by their corner classes; once dragged they switch to
// fixed left/top coordinates so they stay where the user put them.
const mainPanel = ref(null)
const logPanel = ref(null)
const statsPanel = ref(null)
const voxelInspector = ref(null)
const entropyInspector = ref(null)
const filamentInspector = ref(null)
const gantryInspector = ref(null)

const dragState = { active: false, moved: false, startX: 0, startY: 0, origLeft: 0, origTop: 0, width: 0 }

const beginDrag = (evt, panelEl) => {
  if (!panelEl || evt.button !== 0) return
  if (evt.target.closest('button')) return // let the close-x button keep working
  dragState.active = true
  dragState.moved = false
  dragState.startX = evt.clientX
  dragState.startY = evt.clientY
  const rect = panelEl.getBoundingClientRect()
  dragState.origLeft = rect.left
  dragState.origTop = rect.top
  dragState.width = rect.width

  const onMove = (e) => {
    if (!dragState.active) return
    const dx = e.clientX - dragState.startX
    const dy = e.clientY - dragState.startY
    if (!dragState.moved && Math.abs(dx) + Math.abs(dy) > 4) {
      dragState.moved = true
      panelEl.style.position = 'fixed'
      panelEl.style.right = 'auto'
      panelEl.style.bottom = 'auto'
      panelEl.style.margin = '0'
    }
    if (!dragState.moved) return
    const maxLeft = Math.max(8, window.innerWidth - dragState.width - 8)
    const left = Math.min(Math.max(8, dragState.origLeft + dx), maxLeft)
    const top = Math.min(Math.max(8, dragState.origTop + dy), window.innerHeight - 48)
    panelEl.style.left = left + 'px'
    panelEl.style.top = top + 'px'
  }
  const onUp = () => {
    dragState.active = false
    window.removeEventListener('mousemove', onMove)
    window.removeEventListener('mouseup', onUp)
    window.removeEventListener('blur', onUp)
    // Clear the drag flag after the click event has had its chance to fire,
    // so a drag never collapses the panel via its title-bar click handler.
    setTimeout(() => { dragState.moved = false }, 0)
  }
  window.addEventListener('mousemove', onMove)
  window.addEventListener('mouseup', onUp)
  window.addEventListener('blur', onUp)
  evt.preventDefault()
}

const handleTitleClick = (evt, fn) => {
  if (dragState.moved) return // a click that follows a drag is not a toggle
  fn()
}

const handleSelectVoxel = (data) => {
  selectedGantry.value = null // Clear portal selection
  selectedFilament.value = null

  // Handle both voxel selections and entropy selections
  if (data.type === 'entropy') {
    selectedVoxel.value = null // Clear voxel selection for entropy
    selectedEntropy.value = data // Set entropy selection
    addLog(`NEURAL INSPECT: ENTROPY CORE | ${data.entropy.toFixed(4)}`)
  } else if (data.type === 'filament') {
    selectedVoxel.value = null
    selectedEntropy.value = null
    selectedFilament.value = data
    addLog(`SYNAPTIC LINK: L${data.layerIndex} | INTENSITY: ${(data.intensity*100).toFixed(1)}%`)
  } else {
    selectedEntropy.value = null // Clear entropy selection for voxel
    selectedVoxel.value = data
    addLog(`NEURAL INSPECT: L${data.layerIndex} | U${data.voxelIndex}`)
  }
}

const handleSelectGantry = (data) => {
  selectedVoxel.value = null // Clear unit selection
  selectedFilament.value = null
  selectedGantry.value = data
  addLog(`PORTAL INSPECT: L${data.layerIndex} | GATING_MANIFOLD`)
}

const handleDeselect = () => {
  selectedVoxel.value = null
  selectedGantry.value = null
  selectedEntropy.value = null
  selectedFilament.value = null
}

// Watch for entropy value updates when inspector is open
watch(() => currentStep.value?.entropy, (newEntropy) => {
  if (selectedEntropy.value && newEntropy !== undefined) {
    selectedEntropy.value.entropy = newEntropy
  }
})

const nextStep = () => {
  if (currentIndex.value < trace.value.length - 1) {
    currentIndex.value++
    const s = trace.value[currentIndex.value]
    addLog(`TRANSITION: "${s.token}"`)
  } else { togglePlay(false) }
}

const togglePlay = (val = !isPlaying.value) => {
  isPlaying.value = val
  if (isPlaying.value) {
    // Base interval 1400ms, adjusted by speed
    // Speed 0.5 = 2800ms (slower), Speed 2.0 = 700ms (faster)
    const interval = 1400 / playbackSpeed.value
    playInterval = setInterval(nextStep, interval)
  } else { clearInterval(playInterval) }
}

// Classic player controls
const stopPlayback = () => {
  isPlaying.value = false
  clearInterval(playInterval)
  currentIndex.value = 0 // Rewind to start
}

const firstStep = () => {
  isPlaying.value = false
  clearInterval(playInterval)
  currentIndex.value = 0
}

const lastStep = () => {
  isPlaying.value = false
  clearInterval(playInterval)
  if (trace.value.length > 0) {
    currentIndex.value = trace.value.length - 1
  }
}

// Persistence functions
const saveState = () => {
  try {
    localStorage.setItem(STORAGE_KEYS.DOI, doiInput.value)
    // Disabled: Don't save trace to localStorage to avoid quota exceeded errors
    // if (trace.value.length > 0) {
    //   localStorage.setItem(STORAGE_KEYS.TRACE, JSON.stringify(trace.value))
    // }
    localStorage.setItem(STORAGE_KEYS.INDEX, currentIndex.value.toString())
    localStorage.setItem(STORAGE_KEYS.SPEED, playbackSpeed.value.toString())
  } catch (err) {
    console.warn('Failed to save state:', err)
  }
}

const loadState = () => {
  try {
    const savedDoi = localStorage.getItem(STORAGE_KEYS.DOI)
    if (savedDoi) doiInput.value = savedDoi
    
    // Disabled: Don't load trace from localStorage (traces are not saved)
    // const savedTrace = localStorage.getItem(STORAGE_KEYS.TRACE)
    // if (savedTrace) {
    //   try {
    //     const parsed = JSON.parse(savedTrace)
    //     if (Array.isArray(parsed)) {
    //       trace.value = parsed
    //       addLog('RESTORED: Previous session trace loaded')
    //     }
    //   } catch {
    //     console.warn('Failed to parse saved trace')
    //   }
    // }
    
    const savedIndex = localStorage.getItem(STORAGE_KEYS.INDEX)
    if (savedIndex) currentIndex.value = parseInt(savedIndex, 10)
    
    const savedSpeed = localStorage.getItem(STORAGE_KEYS.SPEED)
    if (savedSpeed) playbackSpeed.value = parseFloat(savedSpeed)
  } catch (err) {
    console.warn('Failed to load state:', err)
  }
}

const prevStep = () => { if (currentIndex.value > 0) currentIndex.value-- }

// Speed control
const adjustSpeed = (delta) => {
  const newSpeed = Math.max(0.25, Math.min(8.0, playbackSpeed.value + delta))
  playbackSpeed.value = Math.round(newSpeed * 100) / 100
  saveState()
  if (isPlaying.value) togglePlay(false)
  setTimeout(() => {
    if (isPlaying.value && trace.value.length > 0) togglePlay(true)
  }, 100)
}

const setSpeed = (speed) => {
  playbackSpeed.value = speed
  saveState()
  if (isPlaying.value) togglePlay(false)
  setTimeout(() => {
    if (isPlaying.value && trace.value.length > 0) togglePlay(true)
  }, 100)
}

// Export functionality
const exportTrace = () => {
  if (!trace.value.length) return
  const exportData = {
    metadata: {
      generated_at: new Date().toISOString(),
      paper: paper.value ? { title: paper.value.title, doi: paper.value.doi } : null,
      prediction: prediction.value,
      total_tokens: trace.value.length
    },
    trace: trace.value
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `echo_trace_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
  addLog(`EXPORTED: ${trace.value.length} tokens`)
}

// Import functionality
const fileInput = ref(null)
const triggerImport = () => {
  fileInput.value?.click()
}

const importTrace = (event) => {
  const file = event.target.files?.[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result)
      // Support both flat trace and wrapped format
      const traceData = data.trace || data
      if (!Array.isArray(traceData)) {
        throw new Error('Invalid trace format')
      }

      trace.value = traceData
      currentIndex.value = 0
      addLog(`IMPORTED: ${traceData.length} tokens from ${file.name}`)
    } catch (err) {
      addErrorLog(`IMPORT_ERROR: ${err.message}`)
    }
  }
  reader.readAsText(file)
  event.target.value = '' // Reset input
}
</script>

<template>
  <div class="scanline-overlay"></div>

  <template v-if="loading">
    <div class="loading-screen">
      <div class="brand">ECHO_SYSTEM_INIT</div>
      <div class="loading-bar"><div class="fill"></div></div>
      <div class="status-text">{{ loadingText }}</div>
    </div>
  </template>

  <template v-else>

    <!-- GLOBAL UI GRID -->
    <div class="ui-container">

      <!-- TOP LEFT: BRAND & MISSION CONTROLS -->
      <div class="corner top-left">
        <div class="glass-panel main-brand" ref="mainPanel" :style="{ maxWidth: '420px' }">
          <div class="hud-title-bar" style="cursor: pointer; justify-content: space-between;" @mousedown="beginDrag($event, mainPanel)" @click="handleTitleClick($event, () => mainCollapsed = !mainCollapsed)">
            <span>SYSTEM_NOMINAL</span>
            <div>
              <span class="pulse-dot" style="margin-right: 8px; display: inline-block;"></span>
              <span>{{ mainCollapsed ? '[+]' : '[-]' }}</span>
            </div>
          </div>
          <div v-show="!mainCollapsed" class="brand-content">
            <h1>{{ config.model_id || 'ECHO-DSRN-CLF' }}</h1>

            <div class="integrated-input-area">
              <button @click="fetchRandomPaper" class="ghost-btn primary random-btn" title="FETCH_RANDOM_PAPER">
                🎲 RANDOM PAPER
              </button>
              <input
                v-model="doiInput"
                type="text"
                placeholder="ENTER_DOI..."
                class="integrated-input"
                @keyup.enter="fetchPaperByDoi(doiInput)"
              >
              <button @click="fetchPaperByDoi(doiInput)" class="integrated-send-btn" title="FETCH_BY_DOI">
                ➤
              </button>
            </div>

            <div v-if="paper" class="paper-title" :title="paper.title">
              <span class="paper-label">CURRENT_PAPER</span>
              <span class="paper-text">{{ paper.title }}</span>
            </div>

            <div class="help-infobox">
              <p><strong>INSTRUCTIONS:</strong> Pick a random paper from OpenAIRE or enter a DOI to trace its research-intent classification. Use the timeline controls to step through tokens, or click on visual elements to inspect specific neural activations.</p>
            </div>
            <!-- Classic Music Player Interface -->
            <!-- Professional Media Controller Interface -->
            <div class="player-controls">
              <button @click="prevStep" class="player-btn" title="Step Backward">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <path d="M9 2L1 7l8 5V2z"/>
                  <path d="M13 2L5 7l8 5V2z"/>
                </svg>
              </button>
              <button @click="togglePlay()" class="player-btn primary" :class="{ active: isPlaying }" title="Play/Pause">
                <svg v-if="!isPlaying" width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <path d="M4 2l8 5-8 5V2z"/>
                </svg>
                <svg v-else width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <rect x="3" y="2" width="3" height="10"/>
                  <rect x="8" y="2" width="3" height="10"/>
                </svg>
              </button>
              <button @click="nextStep" class="player-btn" title="Step Forward">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <path d="M5 2l8 5-8 5V2z"/>
                  <path d="M1 2l8 5-8 5V2z"/>
                </svg>
              </button>
              <button @click="stopPlayback" class="player-btn" title="Stop & Rewind">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                  <rect x="2" y="2" width="10" height="10" rx="1"/>
                </svg>
              </button>
            </div>

            <!-- Timeline Scrubber -->
            <div v-if="trace.length > 0" class="timeline-container">
              <input 
                type="range" 
                min="0" 
                :max="trace.length - 1" 
                v-model.number="currentIndex"
                class="timeline-scrubber"
                @mousedown="isPlaying ? togglePlay(false) : null"
              >
              <div class="timeline-labels">
                <span>0</span>
                <span class="active-token">"{{ cleanToken(currentStep?.token) }}"</span>
                <span>{{ trace.length - 1 }}</span>
              </div>
            </div>
            <div class="speed-controls">
              <span class="speed-label">SPEED</span>
              <button @click="adjustSpeed(-0.5)" class="speed-btn">-</button>
              <span class="speed-display">{{ playbackSpeed.toFixed(1) }}x</span>
              <button @click="adjustSpeed(0.5)" class="speed-btn">+</button>
              <button @click="setSpeed(1.0)" class="ghost-btn" style="margin-left: 8px; padding: 4px 8px;">RESET</button>
            </div>
            <div class="data-controls">
              <button @click="exportTrace" class="ghost-btn" :disabled="!trace.length">SAVE TELEMETRY</button>
              <button @click="triggerImport" class="ghost-btn">LOAD TELEMETRY</button>
              <input ref="fileInput" type="file" accept=".json" style="display:none" @change="importTrace">
            </div>
            
            <div class="signature" style="text-align: center; margin-top: 15px; margin-bottom: 0px;">{{ footerText }}</div>
          </div>
        </div>
      </div>

      <!-- BOTTOM LEFT: NARRATIVE TICKER -->
      <div class="corner bottom-left">
        <div class="glass-panel log-container" ref="logPanel" :style="{ height: logCollapsed ? 'auto' : '280px' }">
          <div class="hud-title-bar" style="cursor: pointer; justify-content: space-between;" @mousedown="beginDrag($event, logPanel)" @click="handleTitleClick($event, () => logCollapsed = !logCollapsed)">
            <span>NEURAL_NARRATIVE</span>
            <span>{{ logCollapsed ? '[+]' : '[-]' }}</span>
          </div>
          <div v-show="!logCollapsed" class="log-scroll">
            <div v-for="(log, i) in logs" :key="i" class="log-row" :style="{ opacity: 1 - i * 0.08 }">
              {{ log }}
            </div>
          </div>
        </div>
      </div>

      <!-- BOTTOM RIGHT: AESTHETIC SCHEMA & STATS -->
      <div class="corner bottom-right">
        <div class="glass-panel stats-container" ref="statsPanel">
          <div class="hud-title-bar" style="cursor: pointer; justify-content: space-between;" @mousedown="beginDrag($event, statsPanel)" @click="handleTitleClick($event, () => statsCollapsed = !statsCollapsed)">
            <span>REALTIME_TELEMETRY</span>
            <span>{{ statsCollapsed ? '[+]' : '[-]' }}</span>
          </div>
          <div v-show="!statsCollapsed">
            <div class="stats-grid">
            <div class="stat-box">
              <span class="stat-label">TOKEN</span>
              <span class="stat-value cyber-cyan">"{{ cleanToken(currentStep?.token) }}"</span>
            </div>
            <div class="stat-box">
              <span class="stat-label">λ_GATE (SURPRISE)</span>
              <span class="stat-value neon-amber">{{ Math.max(...(currentStep?.layers?.map(l => l.lambda) || [0])).toFixed(4) }}</span>
            </div>
            <div class="stat-box">
              <span class="stat-label">PRED_INTENT</span>
              <span class="stat-value cyber-cyan" style="font-size: 0.9rem;">{{ predictedIntent }}</span>
            </div>
          </div>
          <div class="legend-grid">
            <div class="legend-item"><span class="swatch grey"></span> λ_GATE</div>
            <div class="legend-item"><span class="swatch cyan"></span> OUTPUT</div>
            <div class="legend-item"><span class="swatch" style="background:#ffaa00;"></span> SYNAPSE</div>
          </div>
          </div>
        </div>
      </div>

    </div>

    <!-- TELEPORTED NEURAL INSPECTOR -->
    <Teleport to="body">
      <!-- VOXEL MODE -->
      <div v-if="selectedVoxel" ref="voxelInspector" class="glass-panel neural-inspector">
        <div class="hud-title-bar" @mousedown="beginDrag($event, voxelInspector)">
          <span>NEURAL_INSPECTOR</span>
          <button @click="selectedVoxel = null" class="close-x">×</button>
        </div>
        <div class="inspector-content">
          <div class="data-row">
            <span class="label">ADDRESS</span>
            <span class="value">L{{ selectedVoxel.layerIndex }} | U{{ selectedVoxel.voxelIndex }}</span>
          </div>
          <div class="data-row highlight">
            <span class="label">SLOW_STATE (C_t)</span>
            <span class="value cyan-glow">{{ selectedVoxelValue.toFixed(6) }}</span>
          </div>
          <div class="activation-bar"><div class="fill" :style="{ width: Math.min(Math.abs(selectedVoxelValue) * 100, 100) + '%' }"></div></div>
          
          <!-- FAST STATE SECTION -->
          <div class="data-row" style="margin-top: 15px;">
            <span class="label" style="color: var(--fast-state-cyan);">FAST_STATE (H_t)</span>
            <span class="value fast-state-cyan">{{ (fastStateVariance * 100).toFixed(1) }}% VAR</span>
          </div>
          <div class="fast-state-chart">
            <div v-for="(bin, i) in fastStateBins" :key="i" class="fast-state-bar" :style="{ height: Math.max(Math.min(bin * 50, 30), 2) + 'px' }"></div>
          </div>

          <div class="architectural-note">
            <p>Discrete dimension within the <strong>$c_t$ (Slow-State)</strong> manifold. This recursive bottleneck distills 2K+ sequence context into O(1) persistent memory. Fast state ($h_t$) renders non-recurrent local activations.</p>
          </div>
        </div>
      </div>

      <!-- ENTROPY MODE -->
      <div v-if="selectedEntropy" ref="entropyInspector" class="glass-panel neural-inspector entropy-mode">
        <div class="hud-title-bar" @mousedown="beginDrag($event, entropyInspector)">
          <span>ENTROPY_CORE</span>
          <button @click="selectedEntropy = null" class="close-x">×</button>
        </div>
        <div class="inspector-content">
          <div class="data-row">
            <span class="label">STATUS</span>
            <span class="value">ACTIVE</span>
          </div>
          <div class="data-row highlight">
            <span class="label">ENTROPY H(P)</span>
            <span class="value magenta-glow">{{ selectedEntropy.entropy.toFixed(4) }}</span>
          </div>
          <div class="activation-bar"><div class="fill" :style="{ width: Math.min(selectedEntropy.entropy * 100, 100) + '%', background: 'var(--accent-alt)' }"></div></div>
          <div class="architectural-note">
            <p><strong>Shannon Entropy:</strong> Measures distributional uncertainty from next-token logits via H(p) = -Σ p(x)log(p(x)). Higher entropy = more uncertainty, Lower entropy = more confident predictions.</p>
          </div>
        </div>
      </div>

      <!-- FILAMENT MODE -->
      <div v-if="selectedFilament" ref="filamentInspector" class="glass-panel neural-inspector filament-mode">
        <div class="hud-title-bar" @mousedown="beginDrag($event, filamentInspector)">
          <span>SYNAPTIC_PATHWAY</span>
          <button @click="selectedFilament = null" class="close-x">×</button>
        </div>
        <div class="inspector-content">
          <div class="data-row">
            <span class="label">SOURCE</span>
            <span class="value">{{ selectedFilamentSource }}</span>
          </div>
          <div class="data-row">
            <span class="label">DESTINATION</span>
            <span class="value">{{ selectedFilament.layerIndex === 'OUT' ? 'OUTPUT HEAD' : 'LAYER ' + selectedFilament.layerIndex }}</span>
          </div>
          <div class="data-row">
            <span class="label">LOCAL LINKS</span>
            <span class="value cyan-glow">{{ selectedFilamentCount }}</span>
          </div>
          <div class="data-row">
            <span class="label">NETWORK TOTAL</span>
            <span class="value cyan-glow">{{ totalNetworkLinks }}</span>
          </div>
          <div class="data-row highlight">
            <span class="label">ATTN INTENSITY</span>
            <span class="value yellow-glow">{{ (selectedFilamentValue * 100).toFixed(1) }}%</span>
          </div>
          <div class="activation-bar"><div class="fill" :style="{ width: Math.min(selectedFilamentValue * 100, 100) + '%', background: '#ffaa00', boxShadow: '0 0 10px rgba(255,170,0,0.5)' }"></div></div>
          <div class="architectural-note">
            <p><strong>Attention Spline:</strong> Visualizes the extrapolated structural linkage derived natively from the sliding window attention and surprise gating parameters.</p>
          </div>
        </div>
      </div>

      <!-- GANTRY MODE -->
      <div v-if="selectedGantry" ref="gantryInspector" class="glass-panel neural-inspector gantry-mode">
        <div class="hud-title-bar" @mousedown="beginDrag($event, gantryInspector)">
          <span>APERTURE_PORTAL</span>
          <button @click="selectedGantry = null" class="close-x">×</button>
        </div>
        <div class="inspector-content">
          <div class="data-row">
            <span class="label">PORTAL_DEPTH</span>
            <span class="value">LAYER_{{ selectedGantry.layerIndex }}</span>
          </div>
          <div class="data-row highlight">
            <span class="label">SURPRISE (λ)</span>
            <span class="value amber-glow">{{ gantryValue.toFixed(4) }}</span>
          </div>
          <div class="activation-bar"><div class="fill" :style="{ width: Math.min(gantryValue * 100, 100) + '%', background: 'var(--accent-alt)' }"></div></div>
          <div class="architectural-note">
            <p>Non-linear recursive gating portal. Controls the flow of gradients into the long-term recurrent bottleneck. High λ indicates a <strong>high-surprise state</strong> triggering substantial aperture shift.</p>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Error Modal -->
    <div v-if="errorState.hasError" class="error-modal" @click.self="resetError">
      <div class="error-content">
        <div class="error-header">
          <span class="error-icon">⚠</span>
          <span class="error-title">SYSTEM_WARNING</span>
        </div>
        <div class="error-message">{{ errorState.message }}</div>
        <div class="error-actions">
          <button @click="lastAction?.()" class="error-btn primary">RETRY</button>
          <button @click="resetError" class="error-btn">DISMISS</button>
        </div>
      </div>
    </div>

    <!-- TelemetryScene -->
    <TelemetryScene
      :trace-step="currentStep"
      :trace-index="currentIndex"
      :num-layers="config.layers"
      :on-log="addLog"
      @selectVoxel="handleSelectVoxel"
      @select-voxel="handleSelectVoxel"
      @selectGantry="handleSelectGantry"
      @select-gantry="handleSelectGantry"
      @deselect="handleDeselect"
    />
  </template>
</template>

<style scoped>
.ui-container { position: fixed; inset: 0; pointer-events: none; padding: 25px; z-index: 100; }
.corner { position: absolute; pointer-events: auto; }
.top-left { top: 25px; left: 25px; }
.bottom-left { bottom: 25px; left: 25px; }
.bottom-right { bottom: 25px; right: 25px; }

.help-infobox { font-size: 0.65rem; color: #888; background: rgba(0,0,0,0.4); padding: 10px; border: 1px solid #333; margin-top: 10px; line-height: 1.4; }
.help-infobox strong { color: var(--accent-alt); }

.main-brand { min-width: 320px; }
.brand-content { padding: 15px; }
h1 { margin: 0 0 15px 0; font-size: 1.1rem; letter-spacing: 0.2rem; color: var(--fg); }
.playback-controls { display: flex; gap: 10px; }
.ghost-btn { background: transparent; border: 1px solid var(--glass-border); color: #888; padding: 6px 12px; font-size: 0.7rem; cursor: pointer; transition: all 0.2s; font-family: var(--font); }
.ghost-btn:hover { border-color: var(--fg); color: var(--fg); }
.ghost-btn.primary { border-color: var(--accent-alt); color: var(--accent-alt); }

.log-container { width: 400px; height: 280px; display: flex; flex-direction: column; }
.log-scroll { flex: 1; overflow-y: auto; padding: 15px; font-size: 0.7rem; line-height: 1.6; }
.log-row { margin-bottom: 5px; color: #aaa; border-left: 2px solid #222; padding-left: 10px; }

.stats-container { width: 300px; padding: 15px; }
.stats-grid { display: flex; flex-direction: column; gap: 15px; margin-bottom: 15px; }
.stat-box { display: flex; justify-content: space-between; align-items: baseline; }
.stat-label { font-size: 0.6rem; color: #666; }
.stat-value { font-size: 1.2rem; font-weight: bold; }
.cyber-cyan { color: var(--accent); }
.neon-amber { color: var(--accent-alt); }

.legend-grid { display: flex; gap: 20px; font-size: 0.6rem; color: #888; padding-top: 15px; border-top: 1px solid var(--glass-border); }
.legend-item { display: flex; align-items: center; gap: 8px; }
.swatch { width: 10px; height: 10px; border: 1px solid rgba(255,255,255,0.1); }
.swatch.grey { background: #666; }
.swatch.cyan { background: var(--accent); }

.neural-inspector { position: fixed; top: 100px; right: 25px; width: 320px; z-index: 10000; }
.inspector-content { padding: 20px; }
.data-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.8rem; }
.data-row.highlight { margin-bottom: 5px; }
.cyan-glow { color: var(--accent); text-shadow: 0 0 10px var(--accent); font-size: 1.4rem; font-weight: bold; }
.amber-glow { color: var(--accent-alt); text-shadow: 0 0 10px var(--accent-alt); font-size: 1.4rem; font-weight: bold; }
.magenta-glow { color: #d946ef; text-shadow: 0 0 10px #d946ef; font-size: 1.4rem; font-weight: bold; }
.activation-bar { height: 3px; background: #222; margin: 10px 0 15px 0; }
.activation-bar .fill { height: 100%; background: var(--accent); box-shadow: 0 0 10px var(--accent); transition: width 0.3s; }
.architectural-note { font-size: 0.65rem; color: #888; line-height: 1.5; border-top: 1px solid #222; padding-top: 15px; }
.close-x { background: none; border: none; color: #fff; font-size: 1.2rem; cursor: pointer; padding: 0; }

.neural-inspector.entropy-mode .activation-bar .fill { background: #d946ef; box-shadow: 0 0 10px #d946ef; }
.neural-inspector.gantry-mode .activation-bar .fill { background: var(--accent-alt); box-shadow: 0 0 10px var(--accent-alt); }

.loading-screen { position: fixed; inset: 0; background: #000; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 10000; }
.loading-bar { width: 200px; height: 2px; background: #111; margin: 20px 0; }
.loading-bar .fill { width: 50%; height: 100%; background: var(--accent-alt); animation: slide 1.5s infinite ease-in-out; }
@keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(200%); } }
.status-text { font-size: 0.75rem; color: #444; letter-spacing: 0.2rem; }

.pulse-dot { width: 8px; height: 8px; background: #00ff00; border-radius: 50%; box-shadow: 0 0 10px #00ff00; animation: pulse-glow 1s infinite; }
@keyframes pulse-glow { from { opacity: 0.3; } to { opacity: 1; } }

/* Integrated Command Styles */
.integrated-input-area { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 15px; }
.integrated-input { flex: 1; background: transparent; border: none; color: var(--fg); font-family: var(--font); font-size: 0.8rem; border-bottom: 1px solid var(--glass-border); padding: 5px 0; transition: all 0.3s; }
.integrated-input:focus { outline: none; border-color: var(--accent); background: rgba(0,255,255,0.02); }
.integrated-send-btn { background: transparent; border: none; color: var(--accent-alt); font-size: 1.1rem; cursor: pointer; transition: transform 0.2s; }
.integrated-send-btn:hover { transform: scale(1.1); color: var(--accent); }

.random-btn { flex: 1; text-align: center; white-space: nowrap; }
.paper-title { margin: 0 0 10px 0; padding: 8px 10px; border: 1px solid rgba(0,255,255,0.15); background: rgba(0,255,255,0.03); }
.paper-label { display: block; font-size: 0.55rem; color: var(--accent); letter-spacing: 0.15rem; margin-bottom: 4px; }
.paper-text { display: block; font-size: 0.72rem; color: var(--fg); line-height: 1.4; max-height: 2.8em; overflow: hidden; }

/* Classic Player Controls */
.player-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.05);
  justify-content: center;
}

.player-btn {
  width: 40px;
  height: 40px;
  background: transparent;
  border: 1px solid var(--glass-border);
  color: var(--fg);
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

.player-btn:hover {
  border-color: var(--accent-alt);
  color: var(--accent-alt);
  background: rgba(255,255,255,0.05);
}

.player-btn.primary {
  border-color: var(--accent);
  color: var(--accent);
}

.player-btn.primary:hover {
  background: var(--accent);
  color: #000;
}

.player-btn.primary.active {
  background: var(--accent);
  color: #000;
  box-shadow: 0 0 10px rgba(0,255,255,0.3);
}

/* Speed Controls */
.speed-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

.speed-label {
  font-size: 0.6rem;
  color: #666;
  letter-spacing: 0.1rem;
}

.speed-btn {
  width: 28px;
  height: 24px;
  background: transparent;
  border: 1px solid var(--glass-border);
  color: var(--fg);
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
  font-family: var(--font);
}

.speed-btn:hover {
  border-color: var(--accent-alt);
  color: var(--accent-alt);
}

.speed-display {
  width: 40px;
  text-align: center;
  font-size: 0.75rem;
  color: var(--accent-alt);
  font-weight: bold;
}

/* Data Controls */
.data-controls { display: flex; gap: 10px; margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); }
.data-controls .ghost-btn { flex: 1; }
.data-controls .ghost-btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* Error Modal */
.error-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(5px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 99999;
  animation: fadeIn 0.2s ease-out;
}

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.error-content {
  background: var(--bg);
  border: 2px solid var(--accent-alt);
  box-shadow: 0 0 30px rgba(255, 174, 0, 0.3);
  padding: 25px;
  min-width: 350px;
  max-width: 500px;
}

.error-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.error-icon { font-size: 1.5rem; }
.error-title { font-size: 0.8rem; letter-spacing: 0.1rem; color: var(--accent-alt); }
.error-message { font-size: 0.85rem; color: #ccc; line-height: 1.6; margin-bottom: 20px; }
.error-actions { display: flex; gap: 10px; }
.error-btn {
  flex: 1;
  padding: 10px;
  font-family: var(--font);
  font-size: 0.75rem;
  letter-spacing: 0.05rem;
  cursor: pointer;
  border: 1px solid var(--glass-border);
  background: rgba(255,255,255,0.05);
  color: var(--fg);
  transition: all 0.2s;
}
.error-btn:hover { border-color: var(--accent-alt); color: var(--accent-alt); }
.error-btn.primary { border-color: var(--accent-alt); color: var(--accent-alt); }
.error-btn.primary:hover { background: var(--accent-alt); color: #000; }
</style>
