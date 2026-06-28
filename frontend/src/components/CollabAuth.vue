<template>
  <div class="glass-card auth-header-card">
    <div v-if="!user.authenticated" class="auth-promo">
      <div class="auth-promo-text">
        <h3>👥 Collaborative Research Intent Labelling</h3>
        <p>
          Join the community to verify model predictions, flag corrupted data,
          and construct a gold-standard dataset for downstream DSRN fine-tuning.
          Every submission is public.
        </p>
      </div>
      <div class="auth-buttons-wrapper">
        <button @click="$emit('login')" class="btn btn-auth btn-hf">
          <span class="auth-icon">🤗</span> Login with Hugging Face
        </button>
      </div>
    </div>
    <div v-else class="auth-profile">
      <div class="profile-info">
        <img
          :src="
            user.avatar_url ||
            'https://api.dicebear.com/7.x/bottts/svg?seed=' + user.name
          "
          alt="Avatar"
          class="avatar"
        />
        <div class="profile-details">
          <div class="name-row">
            <span class="user-name">{{ user.name }}</span>
            <span class="provider-badge">{{
              user.id.split("|")[0].toUpperCase()
            }}</span>
          </div>
          <span class="institution-text">{{
            user.institution || "Academic/Independent"
          }}</span>
        </div>
      </div>
      <div class="profile-actions">
        <button
          @click="$emit('toggle-editor')"
          class="btn btn-secondary btn-sm"
        >
          {{ isEditingProfile ? "Close Settings" : "Edit Profile" }}
        </button>
        <button @click="$emit('logout')" class="btn btn-danger btn-sm">
          Logout
        </button>
      </div>
    </div>

    <div
      v-if="user.authenticated && isEditingProfile"
      class="profile-editor-drawer"
    >
      <h4>Update Registry Information</h4>
      <div class="editor-grid">
        <div class="input-group">
          <label>First Name</label>
          <input
            v-model="localForm.first_name"
            type="text"
            class="cyber-input"
          />
        </div>
        <div class="input-group">
          <label>Last Name</label>
          <input
            v-model="localForm.last_name"
            type="text"
            class="cyber-input"
          />
        </div>
        <div class="input-group">
          <label>Institution/Organisation</label>
          <input
            v-model="localForm.institution"
            type="text"
            class="cyber-input"
          />
        </div>
      </div>
      <div class="editor-actions">
        <button @click="emitSave" class="btn btn-accent btn-sm">
          Save Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch } from "vue";

export default {
  props: {
    user: { type: Object, required: true },
    isEditingProfile: { type: Boolean, default: false },
    profileForm: {
      type: Object,
      default: () => ({ first_name: "", last_name: "", institution: "" }),
    },
  },
  emits: ["login", "logout", "toggle-editor", "update-profile"],
  setup(props, { emit }) {
    const localForm = ref({ ...props.profileForm });

    watch(
      () => props.profileForm,
      (val) => {
        localForm.value = { ...val };
      },
      { immediate: true },
    );

    const emitSave = () => {
      emit("update-profile", { ...localForm.value });
    };

    return { localForm, emitSave };
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
.auth-header-card {
  padding: 1.5rem;
}
.auth-promo {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 2rem;
  flex-wrap: wrap;
}
.auth-promo-text h3 {
  color: var(--accent-blue);
  margin-bottom: 0.5rem;
}
.auth-promo-text p {
  color: var(--text-secondary);
  font-size: 0.95rem;
  max-width: 750px;
}
.auth-buttons-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.btn-auth {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  font-weight: 600;
  padding: 0.5rem 1rem;
}
.btn-hf {
  background: #f1c40f;
  color: #1e293b;
}
.auth-profile {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.profile-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: 2px solid var(--accent-blue);
  background: rgba(255, 255, 255, 0.1);
}
.profile-details {
  display: flex;
  flex-direction: column;
}
.name-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.user-name {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 1.1rem;
}
.provider-badge {
  font-size: 0.7rem;
  font-weight: bold;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.15);
  color: var(--text-secondary);
}
.institution-text {
  color: var(--text-muted);
  font-size: 0.85rem;
}
.profile-actions {
  display: flex;
  gap: 0.5rem;
}
.profile-editor-drawer {
  margin-top: 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  padding-top: 1.25rem;
}
.profile-editor-drawer h4 {
  color: var(--text-primary);
  margin-bottom: 1rem;
  font-family: "Space Grotesk", sans-serif;
}
.editor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}
.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.input-group label {
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-weight: 600;
}
.editor-actions {
  display: flex;
  justify-content: flex-end;
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
.btn-sm {
  font-size: 0.8rem;
  padding: 0.4rem 0.8rem;
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
.btn-secondary {
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.1);
  color: var(--text-primary);
}
.btn-secondary:hover {
  background: rgba(255, 255, 255, 0.1);
}
.btn-danger {
  background: rgba(239, 68, 68, 0.15);
  border-color: rgba(239, 68, 68, 0.2);
  color: #fca5a5;
}
.btn-danger:hover {
  background: rgba(239, 68, 68, 0.25);
}

@media (max-width: 640px) {
  .auth-profile {
    flex-direction: column;
    align-items: stretch;
    gap: 1rem;
  }
  .profile-info {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }
  .profile-details {
    align-items: center;
  }
  .name-row {
    justify-content: center;
  }
  .profile-actions {
    justify-content: center;
    flex-wrap: wrap;
  }
  .profile-actions button {
    flex: 1 1 auto;
  }
  .auth-promo {
    flex-direction: column;
    align-items: stretch;
    gap: 1.25rem;
  }
  .auth-promo-text {
    text-align: center;
  }
  .auth-buttons-wrapper {
    align-items: center;
  }
}
</style>
