# Firebase Security Rules Configuration
# This manages Firebase rules alongside your infrastructure

# Note: Firebase rules are best managed through the Firebase CLI
# since Terraform's Firebase provider has limitations with rules deployment.
# However, we can reference and validate the rules files exist.

# Note: Firestore database already exists, we'll just reference the rules files
# Validate that rules files exist
data "local_file" "firestore_rules" {
  filename = "${path.module}/../frontend/frontend/firestore.rules"
}

data "local_file" "storage_rules" {
  filename = "${path.module}/../frontend/frontend/storage.rules"
}

# Create a deployment script for rules
resource "local_file" "deploy_rules_script" {
  content = <<-EOT
#!/bin/bash
# Firebase Rules Deployment Script
# This script deploys Firebase security rules

set -e

echo "🔒 Deploying Firebase security rules..."

# Change to frontend directory where firebase.json is located
cd "${path.module}/../frontend/frontend"

# Deploy Firestore rules
echo "📄 Deploying Firestore rules..."
firebase deploy --only firestore:rules --project ${var.project_id}

# Deploy Storage rules
echo "🗄️ Deploying Storage rules..."
firebase deploy --only storage --project ${var.project_id}

echo "✅ Firebase rules deployed successfully!"
EOT
  filename = "${path.module}/deploy-rules.sh"
  file_permission = "0755"
}

# Output information about rules management
output "rules_deployment_info" {
  description = "Information about Firebase rules deployment"
  value = {
    firestore_rules_path = data.local_file.firestore_rules.filename
    storage_rules_path   = data.local_file.storage_rules.filename
    deployment_script    = local_file.deploy_rules_script.filename
    manual_deployment    = "Run 'firebase deploy --only firestore:rules,storage' from frontend/frontend directory"
  }
}
