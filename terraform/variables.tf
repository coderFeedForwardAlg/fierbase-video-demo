# Variables for Firebase Video Demo Terraform configuration

variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "qubit-92fe9"
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "firebase_site_id" {
  description = "Firebase Hosting site ID"
  type        = string
  default     = "qubit-92fe9"
}

variable "firestore_location" {
  description = "Firestore database location"
  type        = string
  default     = "nam5"
}

variable "gemini_api_key" {
  description = "Gemini API key for video analysis (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}
