# Production Environment Configuration
# Phase 4 Step 3: Configuration Management & Environment Automation

# Environment Configuration
environment    = "prod"
project_name   = "market-data-agent"
project_owner  = "platform-team"
cost_center    = "engineering"

# AWS Configuration
aws_region = "us-west-2"
vpc_cidr   = "10.2.0.0/16"

# Domain Configuration
domain_name = "api.marketdata.example.com"

# EKS Configuration
kubernetes_version              = "1.28"
cluster_endpoint_public_access  = false  # Private access for production

# Node Group Configuration - Production sizing
node_instance_types     = ["t3.large", "t3.xlarge", "c5.large"]
node_group_min_size     = 2
node_group_max_size     = 20
node_group_desired_size = 5

# Monitoring Node Group - Dedicated monitoring nodes
monitoring_node_instance_types     = ["t3.large"]
monitoring_node_group_min_size     = 2
monitoring_node_group_max_size     = 5
monitoring_node_group_desired_size = 3

# Database Configuration - Production grade
postgres_version        = "15.4"
rds_instance_class      = "db.r5.large"
rds_allocated_storage   = 200
rds_max_allocated_storage = 1000
rds_backup_retention_period = 30
rds_backup_window       = "03:00-04:00"
rds_maintenance_window  = "sun:04:00-sun:05:00"

# Redis Configuration - Production grade
redis_node_type           = "cache.r5.large"
redis_parameter_group     = "default.redis7"
redis_num_nodes          = 3  # Multi-AZ for HA
redis_snapshot_retention = 30
redis_snapshot_window    = "03:00-05:00"
redis_maintenance_window = "sun:05:00-sun:07:00"

# Security Configuration - Restricted access
allowed_cidr_blocks = [
  "10.0.0.0/8",    # Internal networks only
  "172.16.0.0/12", # Corporate networks
  "192.168.0.0/16" # VPN networks
]

# Feature Flags - Full production features
enable_monitoring_stack           = true
enable_backup_automation         = true
enable_ssl_termination          = true
enable_waf                      = true   # Enable WAF for production
enable_detailed_monitoring      = true
enable_vpc_flow_logs           = true
enable_cluster_autoscaler      = true
enable_horizontal_pod_autoscaler = true
enable_vertical_pod_autoscaler  = true
enable_config_rules            = true
enable_guardduty               = true
enable_cloudtrail              = true
enable_cross_region_backup     = true   # Cross-region backup for DR
enable_spot_instances          = false  # On-demand only for production

# Backup Region for DR
backup_region = "us-east-1"

# Monitoring Configuration
monitoring_retention_days = 30

# Database Configuration
database_name     = "marketdata_prod"
database_username = "marketdata_user"

# Environment-specific overrides
environment_config = {
  node_instance_types     = ["t3.large", "t3.xlarge", "c5.large"]
  node_group_min_size     = 2
  node_group_max_size     = 20
  node_group_desired_size = 5
  rds_instance_class      = "db.r5.large"
  redis_node_type         = "cache.r5.large"
  backup_retention_days   = 30
}

# Cluster Users - Production access control
cluster_users = [
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:role/ProductionAdminRole"
  #   username = "prod-admin"
  #   groups   = ["system:masters"]
  # },
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:role/ProductionDeployerRole"
  #   username = "prod-deployer"
  #   groups   = ["deployment"]
  # },
  # {
  #   userarn  = "arn:aws:iam::ACCOUNT:role/ProductionReadOnlyRole"
  #   username = "prod-readonly"
  #   groups   = ["readonly"]
  # }
]

# Admin ARN (replace with your production admin role ARN)
cluster_admin_arn = "arn:aws:iam::ACCOUNT:role/ProductionAdminRole"

# API Keys (set via AWS Systems Manager Parameter Store in production)
yfinance_api_key      = ""  # Retrieved from SSM Parameter Store
alpha_vantage_api_key = ""  # Retrieved from SSM Parameter Store

# Production-specific node group taints
node_group_taints = [
  {
    key    = "environment"
    value  = "production"
    effect = "NO_SCHEDULE"
  }
]