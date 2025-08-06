#!/bin/bash
# Script to deploy the FastAPI application to Google Cloud Run

# Set variables
PROJECT_ID="video-vault-gnytb"  # This should match your Firebase project ID
IMAGE_NAME="video-vault-api"
REGION="us-central1"  # Default region, can be changed
SERVICE_NAME="video-vault-api"

echo "=== Video Vault API Deployment Script ==="
echo "This script will deploy your FastAPI application to Google Cloud Run."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install it first."
    exit 1
fi

# Authenticate with Google Cloud (if needed)
echo "=== Authenticating with Google Cloud ==="
gcloud auth login

# Set the project
echo "=== Setting GCP project ==="
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "=== Enabling required APIs ==="
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and tag the Docker image for Google Container Registry
echo "=== Building and tagging Docker image ==="
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Configure Docker to use gcloud as a credential helper
echo "=== Configuring Docker authentication ==="
gcloud auth configure-docker

# Push the image to Google Container Registry
echo "=== Pushing image to Google Container Registry ==="
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
echo "=== Deploying to Cloud Run ==="
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated

# Get the URL of the deployed service
echo "=== Deployment Complete ==="
echo "Your API is now deployed to Cloud Run!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Service URL: $SERVICE_URL"
echo ""
echo "You can test your API with:"
echo "curl $SERVICE_URL"
echo "curl $SERVICE_URL/ping"
echo ""
echo "To connect your frontend to this API, update your React code to use this URL."
