#!/bin/bash

# Hugging Face Docker Deployment Script
# Usage: ./deploy_to_hf.sh [username] [space-name]

set -e

# Default values
USERNAME=${1:-"your-username"}
SPACE_NAME=${2:-"pdf-bot-api"}

echo "🚀 Deploying to Hugging Face Docker..."
echo "Username: $USERNAME"
echo "Space Name: $SPACE_NAME"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t pdf-bot-api .

# Tag the image for Hugging Face
echo "🏷️  Tagging image for Hugging Face..."
docker tag pdf-bot-api:latest registry.hf.space/$USERNAME/$SPACE_NAME:latest

# Login to Hugging Face (if not already logged in)
echo "🔐 Logging into Hugging Face..."
docker login registry.hf.space

# Push to Hugging Face
echo "⬆️  Pushing to Hugging Face..."
docker push registry.hf.space/$USERNAME/$SPACE_NAME:latest

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://huggingface.co/spaces/$USERNAME/$SPACE_NAME"
echo "2. Make sure the Space is set to 'Docker' SDK"
echo "3. Set the Docker image to: registry.hf.space/$USERNAME/$SPACE_NAME:latest"
echo "4. Add environment variables:"
echo "   - GEMINI_API_KEY: Your Google Gemini API key"
echo "   - AUTHORIZE_TOKEN: Your authentication token"
echo "5. Deploy the Space"
echo ""
echo "🌐 Your API will be available at: https://$USERNAME-$SPACE_NAME.hf.space" 