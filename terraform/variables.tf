# Terraform Variables for Market Data Agent Infrastructure
# Phase 4 Step 3: Configuration Management & Environment Automation

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "market-data-agent"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_owner" {
  description = "Owner of the project"
  type        = string
  default     = "platform-team"
}

variable "cost_center" {
  description = "Cost center for resource allocation"
  type        = string
  default     = "engineering"
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access" {
  description = "Enable public access to EKS cluster endpoint"
  type        = bool
  default     = true
}

variable "cluster_admin_arn" {
  description = "ARN of the user/role that will have admin access to the cluster"
  type        = string
  default     = ""
}

variable "cluster_users" {
  description = "List of additional users to grant access to the cluster"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

# Node Group Configuration
variable "node_instance_types" {
  description = "Instance types for the main node group"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "node_group_min_size" {
  description = "Minimum size of the main node group"
  type        = number
  default     = 1
}

variable "node_group_max_size" {
  description = "Maximum size of the main node group"
  type        = number
  default     = 10
}

variable "node_group_desired_size" {
  description = "Desired size of the main node group"
  type        = number
  default     = 3
}

variable "node_group_taints" {
  description = "Taints to apply to the main node group"
  type = list(object({
    key    = string
    value  = string
    effect = string
  }))
  default = []
}

# Monitoring Node Group Configuration
variable "monitoring_node_instance_types" {
  description = "Instance types for the monitoring node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "monitoring_node_group_min_size" {
  description = "Minimum size of the monitoring node group"
  type        = number
  default     = 1
}

variable "monitoring_node_group_max_size" {
  description = "Maximum size of the monitoring node group"
  type        = number
  default     = 5
}

variable "monitoring_node_group_desired_size" {
  description = "Desired size of the monitoring node group"
  type        = number
  default     = 2
}

# RDS Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "Initial storage allocation for RDS"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "Maximum storage allocation for RDS"
  type        = number
  default     = 100
}

variable "rds_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "rds_backup_window" {
  description = "Backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "rds_maintenance_window" {
  description = "Maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "marketdata"
}

variable "database_username" {
  description = "Database username"
  type        = string
  default     = "marketdata_user"
}

# Redis Configuration
variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_parameter_group" {
  description = "Redis parameter group"
  type        = string
  default     = "default.redis7"
}

variable "redis_num_nodes" {
  description = "Number of Redis nodes"
  type        = number
  default     = 1
}

variable "redis_snapshot_retention" {
  description = "Redis snapshot retention period in days"
  type        = number
  default     = 5
}

variable "redis_snapshot_window" {
  description = "Redis snapshot window"
  type        = string
  default     = "03:00-05:00"
}

variable "redis_maintenance_window" {
  description = "Redis maintenance window"
  type        = string
  default     = "sun:05:00-sun:07:00"
}

# Domain Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# API Keys (should be provided via environment variables or secure parameter store)
variable "yfinance_api_key" {
  description = "Yahoo Finance API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key"
  type        = string
  default     = ""
  sensitive   = true
}

# Environment-specific overrides
variable "environment_config" {
  description = "Environment-specific configuration overrides"
  type = object({
    node_instance_types     = optional(list(string))
    node_group_min_size     = optional(number)
    node_group_max_size     = optional(number)
    node_group_desired_size = optional(number)
    rds_instance_class      = optional(string)
    redis_node_type         = optional(string)
    backup_retention_days   = optional(number)
  })
  default = {}
}

# Feature flags
variable "enable_monitoring_stack" {
  description = "Enable monitoring stack deployment"
  type        = bool
  default     = true
}

variable "enable_backup_automation" {
  description = "Enable automated backup systems"
  type        = bool
  default     = true
}

variable "enable_ssl_termination" {
  description = "Enable SSL termination at load balancer"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = false
}

# Monitoring Configuration
variable "monitoring_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 14
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

# Scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable vertical pod autoscaler"
  type        = bool
  default     = false
}

# Compliance and Governance
variable "enable_config_rules" {
  description = "Enable AWS Config rules for compliance"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backup replication"
  type        = string
  default     = "us-east-1"
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["t3.medium", "t3.large", "c5.large"]
}

# Local variables for computed values
locals {
  # Environment-specific configurations
  env_config = {
    dev = {
      node_instance_types     = ["t3.small", "t3.medium"]
      node_group_min_size     = 1
      node_group_max_size     = 3
      node_group_desired_size = 1
      rds_instance_class      = "db.t3.micro"
      redis_node_type         = "cache.t3.micro"
      backup_retention_days   = 3
    }
    staging = {
      node_instance_types     = ["t3.medium", "t3.large"]
      node_group_min_size     = 1
      node_group_max_size     = 5
      node_group_desired_size = 2
      rds_instance_class      = "db.t3.small"
      redis_node_type         = "cache.t3.small"
      backup_retention_days   = 7
    }
    prod = {
      node_instance_types     = ["t3.large", "t3.xlarge", "c5.large"]
      node_group_min_size     = 2
      node_group_max_size     = 20
      node_group_desired_size = 5
      rds_instance_class      = "db.t3.medium"
      redis_node_type         = "cache.t3.medium"
      backup_retention_days   = 30
    }
  }

  # Merged configuration
  final_config = merge(
    local.env_config[var.environment],
    var.environment_config
  )
}