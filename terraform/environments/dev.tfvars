# Development Environment Configuration
# Phase 4 Step 3: Configuration Management & Environment Automation

# Environment Configuration
environment    = "dev"
project_name   = "market-data-agent"
project_owner  = "platform-team"
cost_center    = "engineering"

# AWS Configuration
aws_region = "us-west-2"
vpc_cidr   = "10.0.0.0/16"

# Domain Configuration (optional for dev)
domain_name = ""

# EKS Configuration
kubernetes_version              = "1.28"
cluster_endpoint_public_access  = true

# Node Group Configuration - Optimized for development
node_instance_types     = ["t3.small", "t3.medium"]
node_group_min_size     = 1
node_group_max_size     = 3
node_group_desired_size = 1

# Monitoring Node Group - Minimal for dev
monitoring_node_instance_types     = ["t3.small"]
monitoring_node_group_min_size     = 1
monitoring_node_group_max_size     = 2
monitoring_node_group_desired_size = 1

# Database Configuration - Cost-optimized for dev
postgres_version        = "15.4"
rds_instance_class      = "db.t3.micro"
rds_allocated_storage   = 20
rds_max_allocated_storage = 50
rds_backup_retention_period = 3
rds_backup_window       = "03:00-04:00"
rds_maintenance_window  = "sun:04:00-sun:05:00"

# Redis Configuration - Minimal for dev
redis_node_type           = "cache.t3.micro"
redis_parameter_group     = "default.redis7"
redis_num_nodes          = 1
redis_snapshot_retention = 3
redis_snapshot_window    = "03:00-05:00"
redis_maintenance_window = "sun:05:00-sun:07:00"

# Security Configuration - More permissive for dev
allowed_cidr_blocks = ["0.0.0.0/0"]

# Feature Flags - Enable all features for testing
enable_monitoring_stack           = true
enable_backup_automation         = true
enable_ssl_termination          = false  # No SSL for dev
enable_waf                      = false  # No WAF for dev
enable_detailed_monitoring      = true
enable_vpc_flow_logs           = true
enable_cluster_autoscaler      = true
enable_horizontal_pod_autoscaler = true
enable_vertical_pod_autoscaler  = false
enable_config_rules            = false  # No compliance rules for dev
enable_guardduty               = false  # No GuardDuty for dev
enable_cloudtrail              = false  # No CloudTrail for dev
enable_cross_region_backup     = false  # No cross-region for dev
enable_spot_instances          = true   # Enable spot for cost savings

# Spot Instance Configuration
spot_instance_types = ["t3.small", "t3.medium"]

# Monitoring Configuration
monitoring_retention_days = 7

# Database Configuration
database_name     = "marketdata_dev"
database_username = "marketdata_user"

# Environment-specific overrides
environment_config = {
  node_instance_types     = ["t3.small", "t3.medium"]
  node_group_min_size     = 1
  node_group_max_size     = 3
  node_group_desired_size = 1
  rds_instance_class      = "db.t3.micro"
  redis_node_type         = "cache.t3.micro"
  backup_retention_days   = 3
}

# Cluster Users (add your ARN here)
cluster_users = [
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:user/developer1"
  #   username = "developer1"
  #   groups   = ["system:masters"]
  # }
]

# Admin ARN (replace with your ARN)
cluster_admin_arn = ""  # Set this to your user/role ARN

# API Keys (set via environment variables)
yfinance_api_key      = ""  # Set via TF_VAR_yfinance_api_key
alpha_vantage_api_key = ""  # Set via TF_VAR_alpha_vantage_api_key