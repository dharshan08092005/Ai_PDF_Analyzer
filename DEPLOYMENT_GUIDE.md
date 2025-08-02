# Render Deployment Guide

## 🚀 Deploy to Render

This guide will help you deploy the optimized PDF Bot API to Render.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Push your code to GitHub
3. **Environment Variables**: Prepare your API keys

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository contains these files:
- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `utils/` - Utility modules
- `render.yaml` - Render configuration (optional)
- `Procfile` - Process definition
- `runtime.txt` - Python version

### 2. Set Up Environment Variables

In your Render dashboard, add these environment variables:

```
GEMINI_API_KEY=your-gemini-api-key-here
AUTHORIZE_TOKEN=your-auth-token-here
FAISS_NO_AVX2=1
FAISS_NO_GPU=1
FAISS_CPU_ONLY=1
```

### 3. Deploy to Render

#### Option A: Using Render Dashboard

1. Go to [render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `pdf-bot-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: Free (or paid for better performance)

#### Option B: Using render.yaml (Blueprints)

1. Push your code with `render.yaml` to GitHub
2. In Render dashboard, click "New +" → "Blueprint"
3. Connect your repository
4. Render will automatically create the service

### 4. Environment Variables Setup

In your Render service dashboard:

1. Go to "Environment" tab
2. Add these variables:
   ```
   GEMINI_API_KEY=your-actual-gemini-key
   AUTHORIZE_TOKEN=your-actual-auth-token
   ```

### 5. Deploy and Test

1. Click "Deploy" in Render
2. Wait for build to complete (usually 2-5 minutes)
3. Your API will be available at: `https://your-app-name.onrender.com`

## API Endpoints

Once deployed, your API will be available at:

- **Health Check**: `GET https://your-app-name.onrender.com/api/v1/health`
- **Main API**: `POST https://your-app-name.onrender.com/api/v1/hackrx/run`

## Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app-name.onrender.com/api/v1/health
```

### 2. Test API
```bash
curl -X POST https://your-app-name.onrender.com/api/v1/hackrx/run \
  -H "Authorization: Bearer your-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## Performance on Render

### Free Tier Limitations:
- **Cold Start**: 30-60 seconds for first request
- **Memory**: 512MB RAM
- **CPU**: Limited
- **Sleep**: Service sleeps after 15 minutes of inactivity

### Paid Tier Benefits:
- **Always On**: No sleep
- **More Resources**: 1GB+ RAM, better CPU
- **Faster Response**: No cold starts

## Troubleshooting

### Common Issues:

1. **Build Failures**
   - Check `requirements.txt` for correct dependencies
   - Ensure Python version compatibility

2. **Import Errors**
   - Verify all dependencies are in `requirements.txt`
   - Check for missing packages

3. **Environment Variables**
   - Ensure `GEMINI_API_KEY` and `AUTHORIZE_TOKEN` are set
   - Check for typos in variable names

4. **Port Issues**
   - Render automatically sets the `PORT` environment variable
   - The code handles this automatically

5. **Memory Issues**
   - Free tier has 512MB RAM limit
   - Consider upgrading for larger documents

### Logs and Debugging:

1. **View Logs**: In Render dashboard → "Logs" tab
2. **Real-time Logs**: Use "Live" tab for real-time debugging
3. **Build Logs**: Check "Build" tab for deployment issues

## Monitoring

### Health Check Endpoint:
```bash
curl https://your-app-name.onrender.com/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "cache_sizes": {
    "document_cache": 5,
    "qa_cache": 23,
    "embedding_cache": 156
  }
}
```

## Cost Optimization

### Free Tier Tips:
- Use smaller documents (under 5MB)
- Implement request caching
- Monitor memory usage
- Consider document preprocessing

### Paid Tier Benefits:
- Better performance
- No cold starts
- More memory for larger documents
- Faster response times

## Security Notes

1. **API Keys**: Never commit API keys to your repository
2. **Environment Variables**: Use Render's secure environment variable system
3. **Authentication**: Always use the `AUTHORIZE_TOKEN` for API access
4. **HTTPS**: Render provides automatic HTTPS

## Support

If you encounter issues:

1. **Check Logs**: Render dashboard → Logs tab
2. **Verify Environment Variables**: Settings → Environment
3. **Test Locally**: Run `python main.py` locally first
4. **Render Support**: Use Render's support system

## Performance Expectations

### Free Tier:
- **Cold Start**: 30-60 seconds
- **Warm Request**: 15-30 seconds
- **Memory**: 512MB limit

### Paid Tier:
- **No Cold Start**: Always ready
- **Response Time**: 15-30 seconds
- **Memory**: 1GB+ available

## Next Steps

1. **Deploy**: Follow the steps above
2. **Test**: Use the health check endpoint
3. **Monitor**: Watch logs for any issues
4. **Optimize**: Consider paid tier for production use
5. **Scale**: Add more resources as needed 