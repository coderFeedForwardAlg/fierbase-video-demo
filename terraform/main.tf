# Terraform configuration for Firebase Video Demo project
# This sets up Firebase Hosting, Cloud Run, and all necessary infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "firebase.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])

  service = each.value
  project = var.project_id

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Create Artifact Registry repository for Docker images
resource "google_artifact_registry_repository" "flask_backend" {
  location      = var.region
  repository_id = "flask-backend"
  description   = "Docker repository for Flask backend"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# Cloud Run service for Flask backend
resource "google_cloud_run_v2_service" "flask_backend" {
  name     = "flask-backend"
  location = var.region
  project  = var.project_id

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/flask-backend/flask-app:latest"
      
      ports {
        container_port = 8080
      }

      # PORT is automatically set by Cloud Run

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "FIREBASE_STORAGE_BUCKET"
        value = "${var.project_id}.appspot.com"
      }

      # Add environment variables for API keys (to be set manually or via Secret Manager)
      dynamic "env" {
        for_each = var.gemini_api_key != "" ? [1] : []
        content {
          name  = "GEMINI_API_KEY"
          value = var.gemini_api_key
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    timeout = "300s"
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.flask_backend
  ]
}

# IAM policy to allow unauthenticated access to Cloud Run service
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_v2_service.flask_backend.name
  location = google_cloud_run_v2_service.flask_backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Create Firebase project (if not already exists)
resource "google_firebase_project" "default" {
  provider = google-beta
  project  = var.project_id

  depends_on = [google_project_service.required_apis]
}

# Enable Firebase Hosting
resource "google_firebase_hosting_site" "default" {
  provider = google-beta
  project  = var.project_id
  site_id  = var.firebase_site_id

  depends_on = [google_firebase_project.default]
}

# Note: Firebase Storage bucket will be created automatically
# when Firebase project is initialized. We don't need to manage it here.

# Note: Firestore database creation is handled in firebase-rules.tf
# to avoid conflicts with rules deployment

# Service account for Cloud Run
resource "google_service_account" "cloud_run_sa" {
  account_id   = "cloud-run-flask-backend"
  display_name = "Cloud Run Flask Backend Service Account"
  description  = "Service account for Flask backend running on Cloud Run"
}

# Grant necessary permissions to the service account
resource "google_project_iam_member" "cloud_run_permissions" {
  for_each = toset([
    "roles/storage.objectAdmin",      # For GCS operations
    "roles/firebase.admin",           # For Firebase operations
    "roles/datastore.user",           # For Firestore operations
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Update Cloud Run service to use the service account
resource "google_cloud_run_v2_service" "flask_backend_with_sa" {
  name     = google_cloud_run_v2_service.flask_backend.name
  location = google_cloud_run_v2_service.flask_backend.location
  project  = var.project_id

  template {
    service_account = google_service_account.cloud_run_sa.email
    
    containers {
      image = google_cloud_run_v2_service.flask_backend.template[0].containers[0].image
      
      ports {
        container_port = 8080
      }

      # PORT is automatically set by Cloud Run

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "FIREBASE_STORAGE_BUCKET"
        value = "${var.project_id}.appspot.com"
      }

      dynamic "env" {
        for_each = var.gemini_api_key != "" ? [1] : []
        content {
          name  = "GEMINI_API_KEY"
          value = var.gemini_api_key
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    timeout = "300s"
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  depends_on = [
    google_service_account.cloud_run_sa,
    google_project_iam_member.cloud_run_permissions
  ]

  lifecycle {
    replace_triggered_by = [
      google_cloud_run_v2_service.flask_backend
    ]
  }
}
