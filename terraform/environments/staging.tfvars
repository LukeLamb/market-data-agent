# Staging Environment Configuration
# Phase 4 Step 3: Configuration Management & Environment Automation

# Environment Configuration
environment    = "staging"
project_name   = "market-data-agent"
project_owner  = "platform-team"
cost_center    = "engineering"

# AWS Configuration
aws_region = "us-west-2"
vpc_cidr   = "10.1.0.0/16"

# Domain Configuration
domain_name = "staging.marketdata.example.com"

# EKS Configuration
kubernetes_version              = "1.28"
cluster_endpoint_public_access  = true

# Node Group Configuration - Moderate sizing for staging
node_instance_types     = ["t3.medium", "t3.large"]
node_group_min_size     = 1
node_group_max_size     = 5
node_group_desired_size = 2

# Monitoring Node Group - Adequate for staging
monitoring_node_instance_types     = ["t3.medium"]
monitoring_node_group_min_size     = 1
monitoring_node_group_max_size     = 3
monitoring_node_group_desired_size = 2

# Database Configuration - Production-like but smaller
postgres_version        = "15.4"
rds_instance_class      = "db.t3.small"
rds_allocated_storage   = 50
rds_max_allocated_storage = 200
rds_backup_retention_period = 7
rds_backup_window       = "03:00-04:00"
rds_maintenance_window  = "sun:04:00-sun:05:00"

# Redis Configuration - Production-like but smaller
redis_node_type           = "cache.t3.small"
redis_parameter_group     = "default.redis7"
redis_num_nodes          = 1
redis_snapshot_retention = 7
redis_snapshot_window    = "03:00-05:00"
redis_maintenance_window = "sun:05:00-sun:07:00"

# Security Configuration - Moderate security
allowed_cidr_blocks = ["0.0.0.0/0"]  # In production, restrict this

# Feature Flags - Production-like features
enable_monitoring_stack           = true
enable_backup_automation         = true
enable_ssl_termination          = true
enable_waf                      = false  # Optional for staging
enable_detailed_monitoring      = true
enable_vpc_flow_logs           = true
enable_cluster_autoscaler      = true
enable_horizontal_pod_autoscaler = true
enable_vertical_pod_autoscaler  = false
enable_config_rules            = true
enable_guardduty               = true
enable_cloudtrail              = true
enable_cross_region_backup     = false  # Optional for staging
enable_spot_instances          = true   # Mix of spot and on-demand

# Spot Instance Configuration
spot_instance_types = ["t3.medium", "t3.large"]

# Monitoring Configuration
monitoring_retention_days = 14

# Database Configuration
database_name     = "marketdata_staging"
database_username = "marketdata_user"

# Environment-specific overrides
environment_config = {
  node_instance_types     = ["t3.medium", "t3.large"]
  node_group_min_size     = 1
  node_group_max_size     = 5
  node_group_desired_size = 2
  rds_instance_class      = "db.t3.small"
  redis_node_type         = "cache.t3.small"
  backup_retention_days   = 7
}

# Cluster Users
cluster_users = [
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:user/qa-team"
  #   username = "qa-team"
  #   groups   = ["system:masters"]
  # },
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:user/staging-deployer"
  #   username = "staging-deployer"
  #   groups   = ["system:masters"]
  # }
]

# Admin ARN (replace with your staging admin ARN)
cluster_admin_arn = ""  # Set this to your staging admin ARN

# API Keys (set via environment variables or parameter store)
yfinance_api_key      = ""  # Set via TF_VAR_yfinance_api_key
alpha_vantage_api_key = ""  # Set via TF_VAR_alpha_vantage_api_key