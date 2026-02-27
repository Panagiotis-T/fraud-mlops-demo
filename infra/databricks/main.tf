terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.0"
    }
  }
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

resource "databricks_job" "fraud_train" {
  name = "fraud-mlops-demo-train"

  task {
    task_key = "train"
    notebook_task {
      notebook_path = "/Repos/<your-user>/fraud-mlops-demo/notebooks/01_train_model"
    }
    existing_cluster_id = var.cluster_id
  }
}