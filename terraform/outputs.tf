# Outputs for Firebase Video Demo Terraform configuration

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "cloud_run_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.flask_backend_with_sa.uri
}

output "firebase_hosting_site_id" {
  description = "Firebase Hosting site ID"
  value       = google_firebase_hosting_site.default.site_id
}

output "firebase_hosting_url" {
  description = "Firebase Hosting URL"
  value       = "https://${google_firebase_hosting_site.default.site_id}.web.app"
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository for Docker images"
  value       = google_artifact_registry_repository.flask_backend.name
}

output "docker_image_url" {
  description = "Full Docker image URL for the backend"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/flask-backend/flask-app:latest"
}

output "service_account_email" {
  description = "Email of the Cloud Run service account"
  value       = google_service_account.cloud_run_sa.email
}

output "firebase_storage_bucket" {
  description = "Firebase Storage bucket name"
  value       = "${var.project_id}.appspot.com"
}
