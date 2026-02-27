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

  job_cluster {
    job_cluster_key = "job-cluster"

    new_cluster {
      spark_version = "13.3.x-scala2.12"
      node_type_id  = "Standard_DS3_v2"
      num_workers   = 1
    }
  }

  task {
    task_key = "train"

    notebook_task {
      notebook_path = "/Repos/panayiwths.ts@gmail.com/fraud-mlops-demo/notebooks/01_train_model"
    }

    job_cluster_key = "job-cluster"
  }
}