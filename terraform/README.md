# Firebase Video Demo - Terraform Infrastructure

This Terraform configuration sets up all the necessary infrastructure for the Firebase Video Demo project, including Firebase Hosting, Cloud Run, and supporting services.

## Architecture

- **Frontend**: Next.js app deployed to Firebase Hosting
- **Backend**: Flask API deployed to Cloud Run
- **Database**: Firestore for data storage
- **Storage**: Firebase Storage for video files
- **Container Registry**: Artifact Registry for Docker images

## Prerequisites

1. **Google Cloud SDK**: Install and authenticate with `gcloud auth login`
2. **Terraform**: Install Terraform >= 1.0
3. **Docker**: For building and pushing container images
4. **Firebase CLI**: For deploying the frontend

## Setup Instructions

### 1. Initialize Terraform

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your actual values
terraform init
```

### 2. Plan and Apply Infrastructure

```bash
# Review the planned changes
terraform plan

# Apply the infrastructure
terraform apply
```

### 3. Build and Deploy Backend

```bash
# Navigate to backend directory
cd ../backend

# Build and push Docker image
gcloud builds submit --tag us-central1-docker.pkg.dev/qubit-92fe9/flask-backend/flask-app:latest

# The Cloud Run service will automatically use the latest image
```

### 4. Deploy Frontend

```bash
# Navigate to frontend directory
cd ../frontend/frontend

# Build and deploy to Firebase Hosting
npm run build
firebase deploy --only hosting
```

## Configuration

### Environment Variables

The following environment variables are automatically configured for the Cloud Run service:

- `PORT`: Set to 8080
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `FIREBASE_STORAGE_BUCKET`: Your Firebase Storage bucket name
- `GEMINI_API_KEY`: (Optional) Set via terraform.tfvars for video analysis

### Firebase Hosting Rewrites

The Firebase Hosting configuration (`firebase.json`) should include these rewrites to route API calls to Cloud Run:

```json
{
  "hosting": {
    "rewrites": [
      {
        "source": "/health",
        "run": {
          "serviceId": "flask-backend",
          "region": "us-central1"
        }
      },
      {
        "source": "/api/**",
        "run": {
          "serviceId": "flask-backend",
          "region": "us-central1"
        }
      }
    ]
  }
}
```

## Permissions

The Terraform configuration creates a service account for Cloud Run with the following permissions:

- `roles/storage.objectAdmin`: For Google Cloud Storage operations
- `roles/firebase.admin`: For Firebase operations
- `roles/firestore.user`: For Firestore database operations

## Monitoring and Logs

- **Cloud Run logs**: Available in Google Cloud Console under Cloud Run
- **Firebase Hosting logs**: Available in Firebase Console
- **Application logs**: Use `gcloud logs tail` to view real-time logs

## Cleanup

To destroy all infrastructure:

```bash
terraform destroy
```

## Troubleshooting

### Common Issues

1. **API not enabled**: The Terraform configuration automatically enables required APIs, but it may take a few minutes to propagate.

2. **Permission denied**: Ensure your gcloud account has the necessary permissions:
   ```bash
   gcloud auth list
   gcloud config set project qubit-92fe9
   ```

3. **Docker image not found**: Make sure to build and push the Docker image before applying Terraform:
   ```bash
   gcloud builds submit --tag us-central1-docker.pkg.dev/qubit-92fe9/flask-backend/flask-app:latest
   ```

4. **Firebase Hosting not routing to Cloud Run**: Verify that the `firebase.json` rewrites configuration matches the Cloud Run service name and region.

### Useful Commands

```bash
# Check Cloud Run service status
gcloud run services list --region=us-central1

# View Cloud Run logs
gcloud logs tail --resource=cloud_run_revision --region=us-central1

# Test the health endpoint
curl https://your-cloud-run-url/health

# Check Firebase Hosting status
firebase hosting:sites:list
```

## Security Notes

- The Cloud Run service is configured to allow unauthenticated access for the API endpoints
- Firebase Authentication is handled within the Flask application
- Sensitive environment variables should be managed through Google Secret Manager for production deployments
- Consider implementing CORS policies based on your frontend domain
