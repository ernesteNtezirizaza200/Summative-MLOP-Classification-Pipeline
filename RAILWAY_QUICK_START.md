# Railway Quick Start Guide

## 🚀 Deploy in 5 Minutes

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for Railway deployment"
git push origin main
```

### Step 2: Deploy on Railway

1. **Go to Railway**: https://railway.app
2. **Sign up/Login** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**: `Summative-MLOP-Classification-Pipeline`
6. **Wait for deployment** (5-15 minutes first time)

### Step 3: Get Your URL

Once deployed, Railway will provide:
- **Public URL**: `https://your-app-name.up.railway.app`
- **API Docs**: `https://your-app-name.up.railway.app/docs`

### Step 4: Test

```bash
# Health check
curl https://your-app-name.up.railway.app/health

# API info
curl https://your-app-name.up.railway.app/api
```

## ✅ That's it!

Your app is now live on Railway. Every push to your main branch will automatically redeploy.

For detailed instructions, see [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)

