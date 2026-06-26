# OAuth — Hugging Face Login

The Collab Hub uses Hugging Face OAuth for authentication. Users log in with their HF account to vote on annotations and save papers.

## Setup

### 1. Create an OAuth App

Go to [Hugging Face OAuth Apps](https://huggingface.co/settings/applications/new) and create a new app:

- **Name**: `Echo-DSRN Collab Hub`
- **Homepage URL**: `http://localhost:7860` (or your production URL)
- **Redirect URL**: `http://localhost:7860/api/auth/callback/hf`

### 2. Configure .env

```env
APP_BASE_URL=http://localhost:7860
HF_CLIENT_ID=<your-client-id>
HF_CLIENT_SECRET=<your-client-secret>
```

### 3. Production

Replace `localhost:7860` with your production domain. Both the Homepage URL and Redirect URL must match.

```env
APP_BASE_URL=https://echo-dsrn.example.com
```

Redirect URL: `https://echo-dsrn.example.com/api/auth/callback/hf`

## How it works

1. User clicks "Login with Hugging Face" in the Collab Hub
2. Redirected to HF authorization page
3. After consent, redirected back to `/api/auth/callback/hf`
4. Backend exchanges code for token, fetches user profile
5. JWT session cookie set — user is authenticated

User data stored: HF username, name, email, avatar URL, and optionally institution (from profile).
