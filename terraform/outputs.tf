# Terraform Outputs for Market Data Agent Infrastructure
# Phase 4 Step 3: Configuration Management & Environment Automation

# Cluster Information
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# Node Groups
output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

# VPC Information
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_arn" {
  description = "ARN of the VPC"
  value       = module.vpc.vpc_arn
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "List of IDs of database subnets"
  value       = module.vpc.database_subnets
}

output "nat_gateway_ids" {
  description = "List of IDs of the NAT Gateways"
  value       = module.vpc.natgw_ids
}

# Database Information
output "db_instance_address" {
  description = "RDS instance hostname"
  value       = module.rds.db_instance_address
  sensitive   = true
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = module.rds.db_instance_arn
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "db_instance_name" {
  description = "RDS instance name"
  value       = module.rds.db_instance_name
}

output "db_instance_username" {
  description = "RDS instance root username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

# Redis Information
output "redis_cluster_address" {
  description = "Redis cluster address"
  value       = module.redis.cluster_address
  sensitive   = true
}

output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = module.redis.cluster_id
}

output "redis_port" {
  description = "Redis port"
  value       = module.redis.port
}

output "redis_cache_nodes" {
  description = "List of cache nodes"
  value       = module.redis.cluster_cache_nodes
  sensitive   = true
}

# Load Balancer Information
output "load_balancer_id" {
  description = "ALB ID"
  value       = module.alb.lb_id
}

output "load_balancer_arn" {
  description = "ALB ARN"
  value       = module.alb.lb_arn
}

output "load_balancer_dns_name" {
  description = "ALB DNS name"
  value       = module.alb.lb_dns_name
}

output "load_balancer_zone_id" {
  description = "ALB zone ID"
  value       = module.alb.lb_zone_id
}

output "target_group_arns" {
  description = "ARNs of the target groups"
  value       = module.alb.target_group_arns
}

# S3 Buckets
output "alb_logs_bucket" {
  description = "ALB logs S3 bucket name"
  value       = aws_s3_bucket.alb_logs.id
}

output "terraform_state_bucket" {
  description = "Terraform state S3 bucket name"
  value       = aws_s3_bucket.terraform_state.id
}

output "market_data_backup_bucket" {
  description = "Market data backup S3 bucket name"
  value       = aws_s3_bucket.market_data_backup.id
}

# IAM Information
output "market_data_agent_role_arn" {
  description = "ARN of the market data agent IAM role"
  value       = aws_iam_role.market_data_agent_role.arn
}

output "market_data_agent_role_name" {
  description = "Name of the market data agent IAM role"
  value       = aws_iam_role.market_data_agent_role.name
}

# Secrets Manager
output "secrets_manager_secret_arn" {
  description = "ARN of the secrets manager secret"
  value       = aws_secretsmanager_secret.market_data_secrets.arn
  sensitive   = true
}

output "secrets_manager_secret_name" {
  description = "Name of the secrets manager secret"
  value       = aws_secretsmanager_secret.market_data_secrets.name
}

# Domain and SSL
output "domain_name" {
  description = "Domain name"
  value       = var.domain_name
}

output "route53_zone_id" {
  description = "Route53 zone ID"
  value       = var.domain_name != "" ? aws_route53_zone.main[0].zone_id : null
}

output "ssl_certificate_arn" {
  description = "SSL certificate ARN"
  value       = var.domain_name != "" ? aws_acm_certificate.main[0].arn : null
}

# Security Groups
output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.redis.id
}

output "alb_security_group_id" {
  description = "ALB security group ID"
  value       = aws_security_group.alb.id
}

# Configuration for kubectl
output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${module.eks.cluster_name}"
}

# Connection strings (for use in application configuration)
output "database_connection_string" {
  description = "Database connection string template"
  value       = "postgresql://${var.database_username}:<password>@${module.rds.db_instance_endpoint}/${var.database_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string template"
  value       = "redis://:<auth_token>@${module.redis.cluster_cache_nodes[0].address}:${module.redis.cluster_cache_nodes[0].port}"
  sensitive   = true
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

# Tags
output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

# Random values (for reference)
output "redis_auth_token" {
  description = "Redis authentication token"
  value       = random_password.redis_auth.result
  sensitive   = true
}

# Monitoring and Observability
output "cloudwatch_log_groups" {
  description = "CloudWatch log groups created"
  value = {
    cluster    = "/aws/eks/${module.eks.cluster_name}/cluster"
    vpc_flow   = module.vpc.vpc_flow_log_cloudwatch_log_group_name
  }
}

# Cost and Resource Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    eks_cluster        = "~$73 (control plane)"
    nodes_t3_large     = "~$67/node/month"
    rds_t3_micro       = "~$16/month"
    redis_t3_micro     = "~$15/month"
    alb                = "~$20/month"
    nat_gateway        = "~$45/month"
    note              = "Actual costs depend on usage, data transfer, and storage"
  }
}

# Deployment Information
output "deployment_info" {
  description = "Information needed for application deployment"
  value = {
    cluster_name              = module.eks.cluster_name
    cluster_endpoint          = module.eks.cluster_endpoint
    oidc_provider_arn         = module.eks.oidc_provider_arn
    service_account_role_arn  = aws_iam_role.market_data_agent_role.arn
    secrets_manager_secret    = aws_secretsmanager_secret.market_data_secrets.name
    load_balancer_target_group = module.alb.target_group_arns[0]
  }
  sensitive = true
}