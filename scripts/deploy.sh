#!/bin/bash

# Firebase Video Demo Deployment Script
# This script deploys both the backend (Cloud Run) and frontend (Firebase Hosting)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="qubit-92fe9"
REGION="us-central1"
SERVICE_NAME="flask-backend"
REPOSITORY="flask-backend"
IMAGE_NAME="flask-app"

echo -e "${BLUE}🚀 Starting deployment for Firebase Video Demo${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set the project
gcloud config set project $PROJECT_ID
print_status "Set project to $PROJECT_ID"

# Deploy infrastructure with Terraform
echo -e "\n${BLUE}📋 Deploying infrastructure with Terraform${NC}"
cd terraform

if [ ! -f "terraform.tfvars" ]; then
    print_warning "terraform.tfvars not found. Creating from example..."
    cp terraform.tfvars.example terraform.tfvars
    print_warning "Please edit terraform.tfvars with your actual values before continuing"
    read -p "Press Enter to continue after editing terraform.tfvars..."
fi

terraform init
terraform plan
read -p "Do you want to apply these changes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply -auto-approve
    print_status "Infrastructure deployed successfully"
else
    print_warning "Terraform apply skipped"
fi

cd ..

# Build and deploy backend
echo -e "\n${BLUE}🐳 Building and deploying backend to Cloud Run${NC}"
cd backend

# Build and submit to Cloud Build
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest

print_status "Backend image built and pushed to Artifact Registry"

# Deploy to Cloud Run (this should happen automatically, but we can trigger an update)
gcloud run deploy $SERVICE_NAME \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300

print_status "Backend deployed to Cloud Run"

# Get the Cloud Run URL
BACKEND_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
echo -e "${GREEN}Backend URL: $BACKEND_URL${NC}"

cd ..

# Deploy frontend
echo -e "\n${BLUE}🌐 Deploying frontend to Firebase Hosting${NC}"
cd frontend/frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm install
fi

# Build the frontend
print_status "Building frontend..."
npm run build

# Deploy to Firebase Hosting
print_status "Deploying to Firebase Hosting..."
firebase deploy --only hosting

print_status "Frontend deployed to Firebase Hosting"

cd ../..

# Test the deployment
echo -e "\n${BLUE}🧪 Testing deployment${NC}"

# Test backend health endpoint
echo "Testing backend health endpoint..."
if curl -f -s "$BACKEND_URL/health" > /dev/null; then
    print_status "Backend health check passed"
else
    print_error "Backend health check failed"
fi

# Get Firebase Hosting URL
FRONTEND_URL="https://$PROJECT_ID.web.app"
echo -e "${GREEN}Frontend URL: $FRONTEND_URL${NC}"

echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}URLs:${NC}"
echo -e "  Frontend: $FRONTEND_URL"
echo -e "  Backend:  $BACKEND_URL"
echo -e "\n${YELLOW}Note: It may take a few minutes for all services to be fully available.${NC}"
