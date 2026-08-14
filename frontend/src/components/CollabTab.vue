<template>
  <div class="collab-container">
    <div v-if="archive" class="archive-banner">
      📦 Read-only archive of round
      <strong>{{ version }}</strong> — annotations are closed for this round.
      <a href="#" @click.prevent="navigateTo('/collab')"
        >View current round {{ currentRound }}</a
      >
    </div>

    <CollabAuth
      v-if="!archive"
      :user="user"
      :isEditingProfile="isEditingProfile"
      :profileForm="profileForm"
      @login="login"
      @logout="logout"
      @toggle-editor="isEditingProfile = !isEditingProfile"
      @update-profile="onProfileUpdate"
    />

    <div class="collab-workspace">
      <CollabCatalog
        :papers="papers"
        :filteredPapers="displayedPapers"
        :totalCount="totalPapers"
        :hasMore="papers.length < totalPapers"
        :isLoading="isLoadingPapers"
        :externalResults="externalResults"
        :searchQuery="searchQuery"
        :searchSource="searchSource"
        :searchPlaceholder="searchPlaceholder"
        :isSearching="isSearching"
        :selectedPaper="selectedPaper"
        :filterLabel="filterLabel"
        :filterSource="filterSource"
        :filterFlagged="filterFlagged"
        :filterAnnotator="filterAnnotator"
        :availableLabels="availableLabels"
        :sortBy="sortBy"
        :readOnly="archive"
        @update:sortBy="onSortChange"
        @update:searchQuery="searchQuery = $event"
        @update:searchSource="onSearchSourceChange($event)"
        @performSearch="performSearch"
        @selectPaper="selectPaper"
        @selectExternalPaper="selectExternalPaper"
        @saveToHistory="saveToHistory"
        @loadMore="loadMore"
        @update:filterLabel="filterLabel = $event"
        @update:filterSource="filterSource = $event"
        @update:filterFlagged="filterFlagged = $event"
        @update:filterAnnotator="filterAnnotator = $event"
      />

      <CollabDetail
        :selectedPaper="selectedPaper"
        :user="user"
        :isClassifying="isClassifying"
        :livePrediction="livePrediction"
        :voteForm="voteForm"
        :availableLabels="availableLabels"
        :hubReadOnly="stats.hub_read_only || archive"
        :archive="archive"
        :version="version"
        @submitVote="submitVote"
        @importAndAnnotate="importAndAnnotate"
        @update:voteForm="voteForm = $event"
      />
    </div>

    <CollabStats :stats="stats" :archiveVersion="version" />
  </div>
</template>

<script>
import { ref, onMounted, computed, watch } from "vue";
import CollabAuth from "./CollabAuth.vue";
import CollabCatalog from "./CollabCatalog.vue";
import CollabDetail from "./CollabDetail.vue";
import CollabStats from "./CollabStats.vue";

export default {
  components: { CollabAuth, CollabCatalog, CollabDetail, CollabStats },
  props: {
    // Active round is '' (the server-configured MODEL_VERSION); a non-empty
    // value switches the tab into a read-only archive of that round.
    version: { type: String, default: "" },
    archive: { type: Boolean, default: false },
  },
  emits: ["save"],
  setup(props, { emit }) {
    const user = ref({
      authenticated: false,
      id: "",
      name: "",
      email: "",
      first_name: "",
      last_name: "",
      institution: "",
      avatar_url: "",
    });
    const isEditingProfile = ref(false);
    const profileForm = ref({ first_name: "", last_name: "", institution: "" });

    const papers = ref([]);
    const selectedPaper = ref(null);
    const searchQuery = ref("");
    const stats = ref({
      total_papers: 0,
      total_annotations: 0,
      human_annotations: 0,
      llm_annotations: 0,
      total_users: 0,
      flagged_papers: 0,
    });
    const totalPapers = ref(0);

    const searchSource = ref("local");
    const isSearching = ref(false);
    const externalResults = ref([]);
    const livePrediction = ref(null);
    const isClassifying = ref(false);
    const voteForm = ref({
      proposed_label: "Methodology",
      is_flagged: false,
      flag_reason: "",
      comment: "",
    });
    const DEFAULT_LABELS = [
      "Methodology",
      "Dataset",
      "Review",
      "Applied",
      "Theoretical",
      "Unclassifiable",
    ];

    // Labels come from the backend (the active model's class set); fall back
    // to the 6-label list when stats have not loaded yet.
    const availableLabels = computed(() =>
      stats.value.labels && stats.value.labels.length
        ? stats.value.labels
        : DEFAULT_LABELS,
    );

    // Round scoping: explicit archive version wins, otherwise the active
    // round reported by the server.
    const activeVersion = computed(
      () => props.version || stats.value.model_version || "",
    );
    const currentRound = computed(
      () => stats.value.model_version || "current round",
    );

    const navigateTo = (path) => {
      window.history.pushState(null, "", path);
      window.dispatchEvent(new PopStateEvent("popstate"));
    };

    // Switching between the active round and an archive does not remount this
    // tab (activeTab stays 'collab') — refetch the versioned data.
    watch(
      () => props.version,
      () => {
        displayCount.value = PAGE_SIZE;
        selectedPaper.value = null;
        livePrediction.value = null;
        externalResults.value = [];
        fetchPapers();
        fetchStats();
      },
    );

    const filterLabel = ref("");
    const filterSource = ref("");
    const filterFlagged = ref(false);
    const filterAnnotator = ref("");
    const sortBy = ref("recent");

    // ── URL query sync ───────────────────────────────────────────────────────
    const readQueryParams = () => {
      const p = new URLSearchParams(window.location.search);
      if (p.has("q")) searchQuery.value = p.get("q") || "";
      if (p.has("label")) filterLabel.value = p.get("label") || "";
      if (p.has("source")) filterSource.value = p.get("source") || "";
      if (p.has("annotator")) filterAnnotator.value = p.get("annotator") || "";
      if (p.has("flagged")) filterFlagged.value = p.get("flagged") === "1";
      if (p.has("sort")) sortBy.value = p.get("sort") || "recent";
      if (p.has("src")) searchSource.value = p.get("src") || "local";
    };

    readQueryParams();

    let _syncTimer = null;
    const syncQueryParams = () => {
      const p = new URLSearchParams();
      if (searchQuery.value) p.set("q", searchQuery.value);
      if (filterLabel.value) p.set("label", filterLabel.value);
      if (filterSource.value) p.set("source", filterSource.value);
      if (filterAnnotator.value) p.set("annotator", filterAnnotator.value);
      if (filterFlagged.value) p.set("flagged", "1");
      if (sortBy.value !== "recent") p.set("sort", sortBy.value);
      if (searchSource.value !== "local") p.set("src", searchSource.value);
      const qs = p.toString();
      const url = window.location.pathname + (qs ? "?" + qs : "");
      window.history.replaceState(null, "", url);
    };

    watch(
      [
        filterLabel,
        filterSource,
        filterFlagged,
        filterAnnotator,
        sortBy,
        searchSource,
      ],
      () => {
        syncQueryParams();
      },
    );
    watch(searchQuery, () => {
      clearTimeout(_syncTimer);
      _syncTimer = setTimeout(syncQueryParams, 400);
    });

    // ── Pagination ────────────────────────────────────────────────────────────
    const PAGE_SIZE = 10;
    const displayCount = ref(PAGE_SIZE);
    const isLoadingPapers = ref(false);

    const filteredPapers = computed(() => {
      let result = papers.value;
      const q = searchQuery.value.trim().toLowerCase();
      if (q) {
        result = result.filter(
          (p) =>
            (p.title || "").toLowerCase().includes(q) ||
            (p.doi || "").toLowerCase().includes(q),
        );
      }
      if (filterFlagged.value) {
        result = result.filter((p) => p.flag_count > 0);
      }
      if (filterAnnotator.value === "human") {
        result = result.filter((p) => p.has_human);
      } else if (filterAnnotator.value === "llm") {
        result = result.filter((p) => p.has_llm);
      }
      return result;
    });

    const displayedPapers = computed(() => {
      return filteredPapers.value.slice(0, displayCount.value);
    });

    // ── Papers fetching ───────────────────────────────────────────────────────
    const fetchPapers = async () => {
      try {
        const params = new URLSearchParams({
          limit: PAGE_SIZE,
          offset: 0,
          sort_by: sortBy.value,
        });
        if (activeVersion.value) params.set("version", activeVersion.value);
        if (filterLabel.value) params.set("label", filterLabel.value);
        if (filterSource.value) params.set("source", filterSource.value);
        const res = await fetch(`/api/annotations/papers?${params}`);
        const data = await res.json();
        papers.value = data.papers;
        totalPapers.value = data.total;
        displayCount.value = Math.min(PAGE_SIZE, data.papers.length);
      } catch (err) {
        console.error("Failed to load papers catalog:", err);
      }
    };

    const loadMorePapers = async () => {
      if (isLoadingPapers.value) return;
      if (papers.value.length >= totalPapers.value) return;
      isLoadingPapers.value = true;
      try {
        const params = new URLSearchParams({
          limit: PAGE_SIZE,
          offset: papers.value.length,
          sort_by: sortBy.value,
        });
        if (activeVersion.value) params.set("version", activeVersion.value);
        if (filterLabel.value) params.set("label", filterLabel.value);
        if (filterSource.value) params.set("source", filterSource.value);
        const res = await fetch(`/api/annotations/papers?${params}`);
        const data = await res.json();
        papers.value = papers.value.concat(data.papers);
        displayCount.value = papers.value.length;
      } catch (err) {
        console.error("Failed to load more:", err);
      } finally {
        isLoadingPapers.value = false;
      }
    };

    const loadMore = () => {
      loadMorePapers();
    };

    // ── Reset pagination on filter/search change ──────────────────────────────
    watch([searchQuery, filterFlagged, filterAnnotator], () => {
      displayCount.value = PAGE_SIZE;
    });
    watch([filterLabel, filterSource], () => {
      displayCount.value = PAGE_SIZE;
      fetchPapers();
    });

    const onSortChange = (val) => {
      sortBy.value = val;
      displayCount.value = PAGE_SIZE;
      fetchPapers();
    };

    const onSearchSourceChange = (val) => {
      // Archives are browse-only: external import is disabled.
      if (props.archive) return;
      searchSource.value = val;
      searchQuery.value = "";
      externalResults.value = [];
    };

    const searchPlaceholder = computed(() => {
      return searchSource.value === "local"
        ? "Search local papers by Title or DOI..."
        : "Search OpenAIRE by publication keywords...";
    });

    // ── Auth ──────────────────────────────────────────────────────────────────
    const fetchUser = async () => {
      try {
        const res = await fetch("/api/auth/me");
        const data = await res.json();
        if (data.authenticated) {
          user.value = data;
          profileForm.value = {
            first_name: data.first_name || "",
            last_name: data.last_name || "",
            institution: data.institution || "Academic/Independent",
          };
        } else {
          user.value = { authenticated: false };
        }
      } catch (err) {
        console.error("Failed to fetch user:", err);
      }
    };

    const login = async () => {
      try {
        const res = await fetch("/api/auth/login/hf");
        const data = await res.json();
        if (data.auth_url) window.location.href = data.auth_url;
      } catch (err) {
        console.error("Login error:", err);
      }
    };

    const logout = async () => {
      try {
        await fetch("/api/auth/logout");
        user.value = { authenticated: false };
        isEditingProfile.value = false;
      } catch (err) {
        console.error("Logout failed:", err);
      }
    };

    const onProfileUpdate = async (form) => {
      try {
        const res = await fetch("/api/auth/profile", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        });
        if (res.ok) {
          await fetchUser();
          isEditingProfile.value = false;
        }
      } catch (err) {
        console.error(err);
      }
    };

    const fetchStats = async () => {
      try {
        const url = activeVersion.value
          ? `/api/annotations/stats?version=${encodeURIComponent(
              activeVersion.value,
            )}`
          : "/api/annotations/stats";
        const res = await fetch(url);
        stats.value = await res.json();
      } catch (err) {
        console.error(err);
      }
    };

    const selectPaper = async (doi) => {
      livePrediction.value = null;
      try {
        const url = activeVersion.value
          ? `/api/annotations/papers/${encodeURIComponent(
              doi,
            )}?version=${encodeURIComponent(activeVersion.value)}`
          : `/api/annotations/papers/${encodeURIComponent(doi)}`;
        const res = await fetch(url);
        if (res.ok) {
          selectedPaper.value = await res.json();
          voteForm.value = {
            proposed_label:
              selectedPaper.value.consensus_label ||
              selectedPaper.value.initial_intent ||
              "Methodology",
            is_flagged: false,
            flag_reason: "",
            comment: "",
          };
          classifyPaper(selectedPaper.value);
        }
      } catch (err) {
        console.error(err);
      }
    };

    const performSearch = async () => {
      const q = searchQuery.value.trim();
      if (!q) {
        externalResults.value = [];
        return;
      }
      if (searchSource.value === "local") return;
      isSearching.value = true;
      externalResults.value = [];
      try {
        const res = await fetch(
          `/api/${searchSource.value}/search?q=${encodeURIComponent(q)}`,
        );
        if (res.ok) externalResults.value = await res.json();
      } catch (err) {
        console.error("Search failed:", err);
      } finally {
        isSearching.value = false;
      }
    };

    const selectExternalPaper = async (extPaper) => {
      try {
        const url = activeVersion.value
          ? `/api/annotations/papers/${encodeURIComponent(
              extPaper.doi,
            )}?version=${encodeURIComponent(activeVersion.value)}`
          : `/api/annotations/papers/${encodeURIComponent(extPaper.doi)}`;
        const res = await fetch(url);
        if (res.ok) {
          const existing = await res.json();
          selectedPaper.value = existing;
          voteForm.value = {
            proposed_label:
              existing.consensus_label ||
              existing.initial_intent ||
              "Methodology",
            is_flagged: false,
            flag_reason: "",
            comment: "",
          };
          classifyPaper(existing);
        } else {
          selectedPaper.value = {
            ...extPaper,
            is_external_pending: true,
            annotations: [],
            consensus_label: null,
          };
          classifyPaper(extPaper);
          voteForm.value = {
            proposed_label: "Methodology",
            is_flagged: false,
            flag_reason: "",
            comment: "",
          };
        }
      } catch (err) {
        console.error(err);
      }
    };

    const classifyPaper = async (paper) => {
      if (!paper || !paper.abstract) return;
      isClassifying.value = true;
      livePrediction.value = null;
      try {
        // min_chars=1: the collab panel should classify short-but-real
        // abstracts (e.g. grant numbers); the 50-char guard stays for the
        // free-text/streaming tabs (default).
        const res = await fetch("/api/classify/intent?min_chars=1", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: paper.title,
            abstract: paper.abstract,
          }),
        });
        if (res.ok) livePrediction.value = await res.json();
      } catch (err) {
        console.error("Classification failed:", err);
      } finally {
        isClassifying.value = false;
      }
    };

    // ── Voting ────────────────────────────────────────────────────────────────
    const submitVote = async () => {
      if (props.archive) return;
      try {
        const res = await fetch("/api/annotations/vote", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            doi: selectedPaper.value.doi,
            proposed_label: voteForm.value.proposed_label,
            is_flagged: voteForm.value.is_flagged,
            flag_reason: voteForm.value.flag_reason,
            comment: voteForm.value.comment,
          }),
        });
        if (res.ok) {
          await selectPaper(selectedPaper.value.doi);
          await fetchPapers();
          await fetchStats();
        }
      } catch (err) {
        console.error(err);
      }
    };

    const importAndAnnotate = async () => {
      if (props.archive) return;
      if (!selectedPaper.value) return;
      try {
        const res = await fetch("/api/annotations/papers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            doi: selectedPaper.value.doi,
            title: selectedPaper.value.title,
            abstract: selectedPaper.value.abstract,
            initial_intent: voteForm.value.proposed_label || "Methodology",
            source: selectedPaper.value.source,
          }),
        });
        if (res.ok) {
          await fetchPapers();
          await submitVote();
        } else {
          const data = await res.json();
          alert(data.detail || "Import failed");
        }
      } catch (err) {
        console.error("Import and vote failed:", err);
      }
    };

    // ── Save to history ───────────────────────────────────────────────────────
    const saveToHistory = (paper) => {
      emit("save", {
        title: paper.title,
        abstract: paper.abstract || "",
        label: paper.consensus_label || paper.initial_intent || "Unclassified",
        doi: paper.doi || null,
        source: paper.source || "unknown",
        timestamp: Date.now(),
      });
    };

    onMounted(async () => {
      await fetchUser();
      await fetchPapers();
      await fetchStats();

      // Archives are read-only — no pending contributions to process.
      if (props.archive) return;

      // Process any pending collab contribution from another tab
      const pendingJson = localStorage.getItem("echo_pending_collab");
      if (pendingJson && user.value.authenticated) {
        localStorage.removeItem("echo_pending_collab");
        try {
          const paper = JSON.parse(pendingJson);
          const doi = paper.doi || `pending-${Date.now()}`;
          const res = await fetch("/api/annotations/papers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              doi,
              title: paper.title,
              abstract: paper.abstract,
              initial_intent: paper.label,
              source: paper.source || "openaire",
            }),
          });
          if (res.ok) {
            await fetchPapers();
            // Auto-select the imported paper and pre-fill the vote
            await selectPaper(doi);
            voteForm.value.proposed_label = paper.label || "Methodology";
          }
        } catch (e) {
          console.error("Pending contribution failed:", e);
        }
      }
    });

    return {
      user,
      isEditingProfile,
      profileForm,
      papers,
      selectedPaper,
      searchQuery,
      stats,
      voteForm,
      filteredPapers,
      displayedPapers,
      displayCount,
      loadMore,
      isLoadingPapers,
      totalPapers,
      sortBy,
      onSortChange,
      availableLabels,
      login,
      logout,
      onProfileUpdate,
      selectPaper,
      submitVote,
      searchSource,
      isSearching,
      externalResults,
      livePrediction,
      isClassifying,
      searchPlaceholder,
      performSearch,
      selectExternalPaper,
      importAndAnnotate,
      saveToHistory,
      onSearchSourceChange,
      filterLabel,
      filterSource,
      filterFlagged,
      filterAnnotator,
      currentRound,
      navigateTo,
    };
  },
};
</script>

<style scoped>
.collab-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.archive-banner {
  background: rgba(234, 179, 8, 0.08);
  border: 1px solid rgba(234, 179, 8, 0.25);
  color: var(--text-secondary);
  border-radius: 8px;
  padding: 0.75rem 1.1rem;
  font-size: 0.9rem;
}
.archive-banner strong {
  color: #facc15;
}
.archive-banner a {
  margin-left: 0.5rem;
  color: var(--accent-blue);
  text-decoration: none;
  border-bottom: 1px dashed rgba(0, 242, 254, 0.4);
}
.collab-workspace {
  display: grid;
  grid-template-columns: 500px 1fr;
  gap: 1.5rem;
  height: 680px;
}

@media (max-width: 1024px) {
  .collab-workspace {
    grid-template-columns: 1fr;
    height: auto;
  }
}
</style>
