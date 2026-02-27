variable "databricks_host" {
  type        = string
  description = "Databricks workspace URL"
}

variable "databricks_token" {
  type        = string
  description = "Databricks personal access token"
}

variable "cluster_id" {
  type        = string
  description = "Existing cluster ID for the job"
}