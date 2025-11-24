# Railway Deployment Guide - Brain Tumor Classification App

This guide will walk you through deploying your Brain Tumor Classification app on Railway.

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] GitHub account
- [ ] Code pushed to a GitHub repository
- [ ] Railway account (sign up at https://railway.app)
- [ ] Dockerfile in your project root (✅ already exists)
- [ ] Model file (`models/brain_tumor_model.h5`) in repository or accessible

---

## Step 1: Prepare Your Code for Deployment

### 1.1 Verify Dockerfile

Your Dockerfile is already configured correctly:
- ✅ Uses `PORT` environment variable
- ✅ Exposes port 8000
- ✅ Has health check endpoint
- ✅ Uses FastAPI with uvicorn

### 1.2 Verify railway.json

The `railway.json` file is already created and configured for Docker deployment.

### 1.3 Commit and Push to GitHub

```bash
# Add all files
git add .

# Commit changes
git commit -m "Prepare for Railway deployment"

# Push to GitHub
git push origin main
```

---

## Step 2: Create Railway Account and Connect GitHub

### 2.1 Sign Up for Railway

1. Go to **https://railway.app**
2. Click **"Start a New Project"** or **"Sign Up"**
3. Sign up using:
   - GitHub account (Recommended - easier integration)
   - Email and password

### 2.2 Connect GitHub Account

If you signed up with email:
1. After signup, go to **Settings** → **GitHub**
2. Click **"Connect GitHub"**
3. Authorize Railway to access your GitHub repositories
4. Select the repositories you want to deploy (or select "All repositories")

---

## Step 3: Create a New Project on Railway

### 3.1 Start Creating Project

1. In Railway dashboard, click **"New Project"** button
2. Select **"Deploy from GitHub repo"**

### 3.2 Select Your Repository

1. You'll see a list of your GitHub repositories
2. Find and click on **"Summative-MLOP-Classification-Pipeline"** (or your repo name)
3. Railway will automatically detect the Dockerfile

---

## Step 4: Configure Deployment Settings

### 4.1 Railway Auto-Detection

Railway will automatically:
- ✅ Detect your Dockerfile
- ✅ Set up the build process
- ✅ Configure the deployment

### 4.2 Environment Variables (Optional)

Railway automatically sets:
- `PORT` - The port your app should listen on (automatically set)

You can add additional environment variables if needed:
1. Go to your service → **Variables** tab
2. Add any custom environment variables:
   - `PYTHONUNBUFFERED=1` (recommended for better logging)
   - `TF_CPP_MIN_LOG_LEVEL=2` (already in Dockerfile, but can override)

### 4.3 Service Settings

1. **Service Name**: Railway will auto-generate one, or you can rename it
2. **Region**: Choose closest to your users (default is usually fine)
3. **Branch**: Select your main branch (usually `main` or `master`)

---

## Step 5: Deploy Your Application

### 5.1 Automatic Deployment

Railway will automatically:
1. Build your Docker image
2. Deploy your application
3. Provide you with a public URL

### 5.2 Monitor the Build Process

You'll see the build logs in real-time:

1. **Building Docker Image**: 
   - Railway pulls your code
   - Builds Docker image using your Dockerfile
   - Installs dependencies
   - This may take 5-15 minutes (first build)

2. **Deploying**:
   - Starts the container
   - Runs your application
   - Health checks

### 5.3 Watch for Errors

Common issues to watch for:
- ❌ **Build fails**: Check Dockerfile syntax
- ❌ **Dependencies fail**: Check requirements.txt
- ❌ **Port binding error**: Ensure Dockerfile uses $PORT (✅ already done)
- ❌ **Model not found**: Verify model file is in repository

---

## Step 6: Verify Deployment

### 6.1 Get Your App URL

Once deployment completes, you'll see:
- **Status**: "Active" (green)
- **URL**: `https://your-app-name.up.railway.app`

### 6.2 Test Your Application

1. Click on the URL or copy it
2. Test the application:
   - ✅ Home page loads (`/`)
   - ✅ API documentation works (`/docs`)
   - ✅ Health check works (`/health`)
   - ✅ Prediction endpoint works (`/predict`)

### 6.3 Check Logs

1. In Railway dashboard, click on your service
2. Go to **"Deployments"** tab
3. Click on the latest deployment
4. Check **"Logs"** for any errors or warnings
5. Verify application started successfully

---

## Step 7: Post-Deployment Configuration (Optional)

### 7.1 Custom Domain (Optional)

1. Go to your service → **Settings** → **Networking**
2. Click **"Generate Domain"** or **"Add Custom Domain"**
3. Follow DNS configuration instructions

### 7.2 Environment Variables Updates

You can add/update environment variables anytime:
1. Go to your service → **Variables** tab
2. Add or modify variables
3. Click **"Save"**
4. App will automatically redeploy

### 7.3 Monitoring

- Check **Metrics** tab for:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

### 7.4 Auto-Deploy Settings

Railway automatically deploys on every push to your connected branch:
- ✅ Enabled by default
- ✅ Can be disabled in service settings if needed

---

## Troubleshooting Common Issues

### Issue 1: Build Fails

**Symptoms**: Build process stops with error

**Solutions**:
1. Check build logs for specific error
2. Verify Dockerfile syntax is correct
3. Ensure all files are in repository
4. Check requirements.txt has all dependencies
5. Verify Python version compatibility
6. Check if model file is too large (use Git LFS if >100MB)

### Issue 2: App Crashes on Startup

**Symptoms**: App builds but crashes when starting

**Solutions**:
1. Check application logs in Railway dashboard
2. Verify model file exists: `models/brain_tumor_model.h5`
3. Check file paths are correct
4. Verify PORT environment variable is used correctly
5. Check memory usage (may need to upgrade plan)
6. Verify all dependencies are installed correctly

### Issue 3: Model Not Loading

**Symptoms**: App starts but model fails to load

**Solutions**:
1. Verify model file is in repository
2. Check model file path in code
3. Ensure model file is not too large (use Git LFS if >100MB)
4. Check file permissions
5. Review error logs for specific error message
6. Verify model file format is correct (.h5)

### Issue 4: Port Binding Error

**Symptoms**: Error about port already in use

**Solutions**:
1. Ensure Dockerfile uses `${PORT}` or `$PORT` environment variable (✅ already done)
2. Don't hardcode port number
3. Railway automatically sets PORT, so your Dockerfile should use it

### Issue 5: Out of Memory

**Symptoms**: App crashes or becomes unresponsive

**Solutions**:
1. Upgrade to a higher plan (Railway offers different resource tiers)
2. Optimize model loading (lazy loading)
3. Reduce batch sizes
4. Use model quantization
5. Check Railway metrics to see memory usage

### Issue 6: Slow Build Times

**Symptoms**: Build takes a very long time

**Solutions**:
1. Optimize Dockerfile layers (already done with requirements.txt first)
2. Use .dockerignore to exclude unnecessary files
3. Consider using Railway's build cache
4. Check if large files are being copied unnecessarily

---

## Quick Reference Commands

### Local Docker Testing (Before Deploying)

Test your Docker setup locally:

```bash
# Build Docker image
docker build -t brain-tumor-app .

# Run Docker container
docker run -p 8000:8000 -e PORT=8000 brain-tumor-app

# Test in browser
# Open http://localhost:8000
# Open http://localhost:8000/docs for API docs
```

### Git Commands

```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

### Railway CLI (Optional)

Install Railway CLI for advanced management:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# View logs
railway logs

# Open in browser
railway open
```

---

## Deployment Checklist

Use this checklist before deploying:

- [ ] Dockerfile exists and is correct
- [ ] Dockerfile uses PORT environment variable (✅ done)
- [ ] railway.json is configured (✅ done)
- [ ] All code is committed to Git
- [ ] Code is pushed to GitHub
- [ ] Model file exists (or download mechanism in place)
- [ ] requirements.txt is complete
- [ ] .dockerignore is configured (optional but recommended)
- [ ] Railway account created
- [ ] GitHub connected to Railway
- [ ] Project created and repository connected
- [ ] Environment variables configured (if needed)
- [ ] Deployment successful
- [ ] Application tested and working

---

## Important Notes

### Railway Free Tier

- **$5 free credit** per month
- **512 MB RAM** default
- **Auto-scaling** based on usage
- **No sleep** - apps stay awake
- **Build time limit**: 90 minutes
- **Database**: Can add PostgreSQL (free tier available)

### Recommendations

1. **For Development**: Free tier is great for testing
2. **For Production**: Consider upgrading for better performance
3. **For Large Models**: Use Git LFS or download from cloud storage
4. **For Database**: Use Railway PostgreSQL (free tier available)

### Railway vs Other Platforms

**Advantages of Railway**:
- ✅ Simple deployment process
- ✅ Automatic HTTPS
- ✅ No sleep (unlike Render free tier)
- ✅ Easy database integration
- ✅ Great developer experience
- ✅ Good free tier

---

## API Endpoints Reference

Once deployed, your API will be available at:

- **Root**: `https://your-app.up.railway.app/`
- **API Info**: `https://your-app.up.railway.app/api`
- **Health Check**: `https://your-app.up.railway.app/health`
- **Uptime**: `https://your-app.up.railway.app/uptime`
- **API Docs**: `https://your-app.up.railway.app/docs`
- **Predict**: `POST https://your-app.up.railway.app/predict`
- **Batch Predict**: `POST https://your-app.up.railway.app/predict/batch`

---

## Support and Resources

- **Railway Documentation**: https://docs.railway.app
- **Railway Status**: https://status.railway.app
- **Docker Documentation**: https://docs.docker.com
- **FastAPI Documentation**: https://fastapi.tiangolo.com

---

## Summary

**Quick Deployment Steps:**
1. Push code to GitHub
2. Create Railway account → Connect GitHub
3. New Project → Select repository
4. Railway auto-detects Dockerfile
5. Configure environment variables (optional)
6. Deploy and test!

Your app will be live at: `https://your-app-name.up.railway.app`

**Congratulations! Your app is live on Railway! 🎉**

