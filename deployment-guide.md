# 🚀 Deployment Guide

## Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI Comment Moderation System"
gh repo create ai-comment-moderation --public --push
# OR
git remote add origin https://github.com/YOUR_USERNAME/ai-comment-moderation.git
git push -u origin main
```

---

## Step 2 — Deploy Backend on Render

1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Set these settings:

| Setting | Value |
|---|---|
| Root directory | `backend` |
| Runtime | Python 3.11 |
| Build command | `pip install -r requirements.txt && python ../scripts/setup_nltk.py` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port 10000` |
| Instance type | Free |

4. Click **Create Web Service**
5. Wait ~3 minutes for the build
6. Copy your backend URL: `https://your-service.onrender.com`

---

## Step 3 — Deploy Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repo
3. Set **Root directory** to `frontend`
4. Add environment variable:

| Key | Value |
|---|---|
| `REACT_APP_API_URL` | `https://your-service.onrender.com` |

5. Click **Deploy**

---

## Step 4 — Connect Them

In `frontend/src/api.js`, ensure:

```js
const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
```

The `REACT_APP_API_URL` env var set in Vercel will be used automatically at build time.

---

## Common Errors

| Error | Fix |
|---|---|
| CORS error in browser | Set the `CORS_ORIGINS` env var on the backend to your exact Vercel URL (no trailing slash); `*` is ignored |
| 503 Models not loaded | Check `/ready`; ensure `backend/models/moderation_model.joblib` is in the repo and NLTK resources were installed at build time |
| 429 Too many requests | Raise `RATE_LIMIT` / `RATE_LIMIT_WINDOW` env vars if legitimate |
| `Missing NLTK resources` | Add `python ../scripts/setup_nltk.py` to the build command |
| Frontend shows "Something went wrong" | Open DevTools → Network → check the response body |
| Render build fails | Check Python version; use `python-3.11.0` in `.python-version` |

---

## Add backend environment variables (Render)

| Key | Value |
|---|---|
| `CORS_ORIGINS` | `https://your-app.vercel.app` |
| `DATABASE_URL` | your Postgres URL (optional; SQLite file is used otherwise and is lost on redeploy) |

## Retraining the model

The repository ships a trained model. To retrain on the public datasets:

```bash
python scripts/download_data.py
python scripts/setup_nltk.py
cd backend && python train.py     # updates models/, experiments/, reports/figures/
```

Commit the regenerated `backend/models/moderation_model.joblib` and `experiments/` files together. See `data/README.md` for dataset sources and licence caveats, and `docs/methodology.md` for the protocol.
