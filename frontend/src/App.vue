<template>
  <div class="app-container">
    <!-- Glowing Accent Background Blobs -->
    <div class="glow-blob glow-left"></div>
    <div class="glow-blob glow-right"></div>

    <!-- Header Section -->
    <header class="header">
      <div class="logo-area">
        <a href="https://www.ethicalabs.ai" target="_blank" class="logo-link">
          <span class="logo-dot"></span>
          <span class="logo-text">ethicalabs.ai</span>
        </a>
        <div class="divider">/</div>
        <div class="app-title">OpenAIRE 2026 — Research Paper Classifier</div>
      </div>
      <div class="subtitle">Research Paper Intent Classification (5-Class)</div>
    </header>

    <!-- Navigation Tab Bar -->
    <nav class="nav-tabs">
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'openaire' }"
        @click="activeTab = 'openaire'"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 17 12 22 22 17"></polyline>
          <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
        OpenAIRE stream
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'collab' }"
        @click="activeTab = 'collab'"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
          <circle cx="9" cy="7" r="4"></circle>
          <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
          <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
        </svg>
        Collab Hub
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'text' }"
        @click="activeTab = 'text'"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path
            d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"
          ></path>
        </svg>
        Free Text
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'saved' }"
        @click="activeTab = 'saved'"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
        </svg>
        Saved History ({{ savedItems.length }})
      </button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'model_card' }"
        @click="activeTab = 'model_card'"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <line x1="9" y1="9" x2="15" y2="9"></line>
          <line x1="9" y1="13" x2="15" y2="13"></line>
          <line x1="9" y1="17" x2="13" y2="17"></line>
        </svg>
        Model Card
      </button>
    </nav>

    <!-- Main Workspace -->
    <main class="workspace">
      <!-- Error Banner -->
      <div v-if="error" class="error-banner">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polygon
            points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"
          ></polygon>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <div class="error-content">
          <span class="error-title">System Alert</span>
          <p class="error-desc">{{ error }}</p>
        </div>
        <button class="close-error" @click="error = ''">&times;</button>
      </div>

      <!-- Health Check Initializing State -->
      <div v-if="isInitializing" class="init-card">
        <div class="cyber-spinner"></div>
        <h3>Initializing Intent Classifier...</h3>
        <p>
          Checking model status on startup. If this is the very first request,
          loading weights takes ~2-3 seconds.
        </p>
        <div class="progress-bar-container">
          <div class="progress-bar-fill"></div>
        </div>
      </div>

      <div v-else>
        <!-- Model Card Tab -->
        <model-card-tab v-if="activeTab === 'model_card'" />

        <!-- OpenAIRE Tab -->
        <openaire-tab
          v-if="activeTab === 'openaire'"
          :is-loading="isLoading"
          @classify="handleClassify"
          @save="handleSave"
          @error="handleError"
        />

        <!-- Custom Text Tab -->
        <text-tab
          v-if="activeTab === 'text'"
          :is-loading="isLoading"
          @classify="handleClassify"
          @save="handleSave"
          @error="handleError"
        />

        <!-- Collab Tab -->
        <collab-tab v-if="activeTab === 'collab'" @save="handleSave" />

        <!-- Saved History Tab -->
        <saved-tab
          v-if="activeTab === 'saved'"
          :items="savedItems"
          @delete="handleDelete"
        />
      </div>
    </main>

    <!-- Footer Area -->
    <footer class="footer">
      <div class="footer-links">
        <a href="https://www.ethicalabs.ai/research/" target="_blank"
          >Research</a
        >
        <a href="https://huggingface.co/ethicalabs" target="_blank"
          >Hugging Face</a
        >
        <a href="https://github.com/ethicalabs-ai" target="_blank">GitHub</a>
      </div>
      <div class="footer-openaire">
        <span
          ><a
            href="https://innovation.openaire.eu/component/content/article/openaire-ai-hackathon.html"
            target="_blank"
            >OpenAIRE AI Hackathon 2026</a
          ></span
        >
        <span>·</span>
        <span
          ><a href="https://graph.openaire.eu" target="_blank"
            >OpenAIRE Graph</a
          ></span
        >
        <span>·</span>
        <span class="cc-by">CC BY 4.0</span>
      </div>
      <div class="footer-legal">
        <div class="copyright">
          &copy; 2026 ethicalabs.ai — incubated by dorvan srl
        </div>
        <div class="address">
          Operational: Via Privata Farnese 1/3, 20146 Milano (MI)
        </div>
        <div class="address">
          Registered: Via Enrico Cernuschi 4, 20129 Milano (MI), Italy
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import { ref, onMounted, watch } from "vue";
import ModelCardTab from "./components/ModelCardTab.vue";
import OpenaireTab from "./components/OpenaireTab.vue";
import TextTab from "./components/TextTab.vue";
import SavedTab from "./components/SavedTab.vue";
import CollabTab from "./components/CollabTab.vue";

export default {
  components: {
    ModelCardTab,
    OpenaireTab,
    TextTab,
    SavedTab,
    CollabTab,
  },
  setup() {
    const activeTab = ref("openaire");
    const isLoading = ref(false);
    const isInitializing = ref(true);
    const error = ref("");
    const savedItems = ref([]);
    const isAuthenticated = ref(false);

    // Load saved items from localStorage
    const loadSavedItems = () => {
      try {
        const stored = localStorage.getItem("echo_intent_history");
        if (stored) {
          savedItems.value = JSON.parse(stored);
        }
      } catch (err) {
        console.error("Failed to load history from localStorage:", err);
      }
    };

    // Sync auth state and merge localStorage with server-side saved papers
    const syncAuthAndHistory = async () => {
      try {
        const res = await fetch("/api/auth/me");
        const data = await res.json();
        if (!data.authenticated) return;
        isAuthenticated.value = true;

        const localItems = JSON.parse(
          localStorage.getItem("echo_intent_history") || "[]",
        );
        if (localItems.length > 0) {
          const syncRes = await fetch("/api/saved/sync", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(localItems),
          });
          if (syncRes.ok) {
            const merged = await syncRes.json();
            savedItems.value = merged;
            localStorage.setItem("echo_intent_history", JSON.stringify(merged));
            return;
          }
        }
        // No local items (or sync failed) — load from DB
        const dbRes = await fetch("/api/saved");
        if (dbRes.ok) {
          const dbItems = await dbRes.json();
          savedItems.value = dbItems;
          localStorage.setItem("echo_intent_history", JSON.stringify(dbItems));
        }
      } catch (err) {
        console.error("Auth sync for saved history failed:", err);
      }
    };

    // Check health status of model on startup
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/classify/intent/health");
        if (res.status === 200) {
          isInitializing.value = false;
        } else {
          error.value =
            "Failed to establish connection to the intent classifier backend.";
          isInitializing.value = false;
        }
      } catch (err) {
        // Retry connection
        setTimeout(checkHealth, 3000);
      }
    };

    const handlePathChange = () => {
      const path = window.location.pathname.replace(/\/$/, "") || "/";
      const tabMap = {
        "/collab": "collab",
        "/model_card": "model_card",
        "/openaire": "openaire",
        "/text": "text",
        "/saved": "saved",
      };
      const tab = tabMap[path];
      if (tab) activeTab.value = tab;
    };

    watch(activeTab, (newTab) => {
      const pathMap = {
        collab: "/collab",
        model_card: "/model_card",
        openaire: "/openaire",
        text: "/text",
        saved: "/saved",
      };
      const path = pathMap[newTab] || "/";
      if (window.location.pathname !== path) {
        window.history.pushState(null, "", path);
      }
    });

    onMounted(() => {
      checkHealth();
      loadSavedItems();
      syncAuthAndHistory();
      handlePathChange();
      window.addEventListener("popstate", handlePathChange);
    });

    // Classify handler
    const handleClassify = async ({ title, abstract, callback }) => {
      isLoading.value = true;
      error.value = "";
      try {
        const response = await fetch("/api/classify/intent", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ title, abstract }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(
            errData.detail || `HTTP error! Status: ${response.status}`,
          );
        }

        const data = await response.json();
        callback(data);
      } catch (err) {
        error.value =
          err.message || "An unexpected error occurred during classification.";
      } finally {
        isLoading.value = false;
      }
    };

    // Save handler
    const handleSave = async (item) => {
      if (savedItems.value.some((i) => i.title === item.title)) return;

      if (isAuthenticated.value) {
        try {
          const res = await fetch("/api/saved", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(item),
          });
          if (res.ok) {
            const saved = await res.json();
            savedItems.value.unshift(saved);
            localStorage.setItem(
              "echo_intent_history",
              JSON.stringify(savedItems.value),
            );
            return;
          }
        } catch (err) {
          console.error("DB save failed, falling back to local:", err);
        }
      }

      savedItems.value.unshift(item);
      try {
        localStorage.setItem(
          "echo_intent_history",
          JSON.stringify(savedItems.value),
        );
      } catch (err) {
        console.error("Failed to persist history to localStorage:", err);
      }
    };

    // Delete handler
    const handleDelete = async (payload) => {
      const idx = typeof payload === "number" ? payload : payload.idx;
      const item =
        typeof payload === "number" ? savedItems.value[idx] : payload.item;
      savedItems.value.splice(idx, 1);
      if (isAuthenticated.value && item && item.id) {
        fetch(`/api/saved/${item.id}`, { method: "DELETE" }).catch(() => {});
      }
      try {
        localStorage.setItem(
          "echo_intent_history",
          JSON.stringify(savedItems.value),
        );
      } catch (err) {
        console.error("Failed to update persisted history:", err);
      }
    };

    const handleError = (msg) => {
      error.value = msg;
    };

    return {
      activeTab,
      isLoading,
      isInitializing,
      error,
      savedItems,
      handleClassify,
      handleSave,
      handleDelete,
      handleError,
    };
  },
};
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  padding: 2.5rem 3rem;
  max-width: 1400px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

/* Accent glowing blobs */
.glow-blob {
  position: fixed;
  width: 500px;
  height: 500px;
  border-radius: 50%;
  filter: blur(140px);
  opacity: 0.05;
  z-index: -1;
  pointer-events: none;
}
.glow-left {
  background: var(--accent-blue);
  top: -10%;
  left: -10%;
}
.glow-right {
  background: var(--accent-purple);
  bottom: -10%;
  right: -10%;
}

.header {
  margin-bottom: 2.5rem;
}
.logo-area {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.logo-link {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-primary);
  font-weight: 500;
  font-size: 1.25rem;
}
.logo-dot {
  width: 8px;
  height: 8px;
  background-color: var(--accent-blue);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--accent-blue);
}
.logo-text {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.01em;
}
.divider {
  color: var(--text-muted);
  font-family: monospace;
}
.app-title {
  color: var(--accent-blue);
  font-weight: 500;
  font-size: 1.25rem;
  font-family: "Space Grotesk", sans-serif;
}
.subtitle {
  color: var(--text-secondary);
  font-size: 0.95rem;
  font-weight: 300;
}

.nav-tabs {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.75rem;
}
.tab-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-secondary);
  font-family: "Space Grotesk", sans-serif;
  font-size: 0.95rem;
  font-weight: 500;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.tab-btn:hover {
  color: var(--text-primary);
  background-color: rgba(255, 255, 255, 0.03);
}
.tab-btn.active {
  color: var(--accent-blue);
  background-color: rgba(0, 242, 254, 0.05);
  border-color: rgba(0, 242, 254, 0.15);
}

.workspace {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.error-banner {
  display: flex;
  align-items: center;
  gap: 1rem;
  background-color: rgba(239, 68, 68, 0.07);
  border: 1px solid rgba(239, 68, 68, 0.2);
  padding: 1rem 1.25rem;
  border-radius: 8px;
  color: #f87171;
  position: relative;
}
.error-content {
  flex: 1;
}
.error-title {
  font-weight: 600;
  font-size: 0.95rem;
  display: block;
  margin-bottom: 0.15rem;
}
.error-desc {
  font-size: 0.9rem;
  opacity: 0.85;
}
.close-error {
  background: transparent;
  border: none;
  color: #f87171;
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0 0.5rem;
}

.init-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 380px;
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 3rem;
  text-align: center;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.init-card h3 {
  color: var(--text-primary);
  margin: 1.5rem 0 0.5rem 0;
  font-size: 1.35rem;
}
.init-card p {
  color: var(--text-secondary);
  font-size: 0.95rem;
  max-width: 480px;
  line-height: 1.5;
  margin-bottom: 2rem;
}
.progress-bar-container {
  width: 100%;
  max-width: 320px;
  height: 4px;
  background-color: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  background: var(--accent-glow);
  width: 30%;
  animation: fillAnim 3s infinite ease-in-out;
}

@keyframes fillAnim {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(300%);
  }
}

.footer {
  margin-top: 3.5rem;
  border-top: 1px solid var(--border-color);
  padding-top: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.footer-links {
  flex: 0 0 auto;
  display: flex;
  gap: 1.5rem;
}
.footer-links a {
  color: var(--text-secondary);
  font-size: 0.9rem;
}
.footer-links a:hover {
  color: var(--accent-blue);
}
.footer-openaire {
  flex: 1 1 auto;
  text-align: center;
  display: flex;
  justify-content: center;
  flex-direction: row;
  gap: 0.5rem;
  align-items: center;
}
.footer-openaire a {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.footer-openaire a:hover {
  color: var(--accent-blue);
}
.footer-legal {
  text-align: right;
  flex: 0 0 auto;
}
.copyright {
  color: var(--text-muted);
  font-size: 0.85rem;
}
.address {
  color: var(--text-muted);
  font-size: 0.7rem;
  opacity: 0.7;
}
.cc-by {
  opacity: 0.5;
  font-style: italic;
}

@media (max-width: 640px) {
  .app-container {
    padding: 1.5rem;
  }
  .footer {
    flex-direction: column;
    text-align: center;
  }
}
</style>
