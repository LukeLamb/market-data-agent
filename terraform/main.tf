# Terraform Main Configuration for Market Data Agent
# Phase 4 Step 3: Configuration Management & Environment Automation

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    vault = {
      source  = "hashicorp/vault"
      version = "~> 3.15"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    # Backend configuration will be provided via backend config file
    # terraform init -backend-config=backend.conf
  }
}

# Configure providers
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "market-data-agent"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.project_owner
      CostCenter  = var.cost_center
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values for computed configurations
locals {
  cluster_name = "${var.project_name}-${var.environment}"

  vpc_cidr = var.vpc_cidr
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)

  private_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 1),
    cidrsubnet(local.vpc_cidr, 4, 2),
    cidrsubnet(local.vpc_cidr, 4, 3),
  ]

  public_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 4),
    cidrsubnet(local.vpc_cidr, 4, 5),
    cidrsubnet(local.vpc_cidr, 4, 6),
  ]

  database_subnets = [
    cidrsubnet(local.vpc_cidr, 4, 7),
    cidrsubnet(local.vpc_cidr, 4, 8),
    cidrsubnet(local.vpc_cidr, 4, 9),
  ]

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.project_owner
    CostCenter  = var.cost_center
  }
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = local.vpc_cidr

  azs              = local.azs
  private_subnets  = local.private_subnets
  public_subnets   = local.public_subnets
  database_subnets = local.database_subnets

  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Database subnet group
  create_database_subnet_group = true
  create_database_subnet_route_table = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  flow_log_max_aggregation_interval    = 60

  # Tags required for EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  tags = local.common_tags
}

# EKS Cluster Module
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = var.cluster_endpoint_public_access

  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      name = "${local.cluster_name}-main"

      instance_types = var.node_instance_types

      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size

      # Launch template configuration
      launch_template_name        = "${local.cluster_name}-main"
      launch_template_description = "Launch template for main node group"

      update_config = {
        max_unavailable_percentage = 25
      }

      # Kubernetes labels
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }

      # Taints for specific workloads if needed
      taints = var.node_group_taints
    }

    monitoring = {
      name = "${local.cluster_name}-monitoring"

      instance_types = var.monitoring_node_instance_types

      min_size     = var.monitoring_node_group_min_size
      max_size     = var.monitoring_node_group_max_size
      desired_size = var.monitoring_node_group_desired_size

      # Launch template configuration
      launch_template_name        = "${local.cluster_name}-monitoring"
      launch_template_description = "Launch template for monitoring node group"

      # Kubernetes labels
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "monitoring"
        Workload    = "monitoring"
      }

      # Taints to ensure only monitoring workloads run on these nodes
      taints = [
        {
          key    = "workload"
          value  = "monitoring"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # Cluster access entries
  access_entries = {
    admin = {
      kubernetes_groups = []
      principal_arn     = var.cluster_admin_arn

      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  # aws-auth ConfigMap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = module.eks.eks_managed_node_groups.main.iam_role_arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups   = ["system:bootstrappers", "system:nodes"]
    },
  ]

  aws_auth_users = var.cluster_users

  tags = local.common_tags
}

# RDS Instance for Market Data Storage
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${local.cluster_name}-postgres"

  # Database configuration
  engine               = "postgres"
  engine_version       = var.postgres_version
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.rds_instance_class

  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_encrypted     = true

  # Database credentials
  db_name  = var.database_name
  username = var.database_username
  manage_master_user_password = true

  # Network configuration
  db_subnet_group_name   = module.vpc.database_subnet_group
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Backup configuration
  backup_retention_period = var.rds_backup_retention_period
  backup_window          = var.rds_backup_window
  maintenance_window     = var.rds_maintenance_window

  # Enhanced monitoring
  monitoring_interval    = 60
  monitoring_role_name   = "${local.cluster_name}-rds-monitoring-role"
  create_monitoring_role = true

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  # Parameter group
  parameters = [
    {
      name  = "log_statement"
      value = "all"
    },
    {
      name  = "log_min_duration_statement"
      value = "1000"
    },
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
  ]

  tags = local.common_tags
}

# ElastiCache Redis Cluster
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id           = "${local.cluster_name}-redis"
  description          = "Redis cluster for market data caching"

  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = var.redis_parameter_group

  num_cache_nodes            = var.redis_num_nodes
  availability_zones         = local.azs

  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]

  # Backup configuration
  snapshot_retention_limit   = var.redis_snapshot_retention
  snapshot_window           = var.redis_snapshot_window

  # Maintenance
  maintenance_window        = var.redis_maintenance_window

  # Enable encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth.result

  tags = local.common_tags
}

# Application Load Balancer
module "alb" {
  source = "terraform-aws-modules/alb/aws"
  version = "~> 8.0"

  name = "${local.cluster_name}-alb"

  load_balancer_type = "application"

  vpc_id             = module.vpc.vpc_id
  subnets            = module.vpc.public_subnets
  security_groups    = [aws_security_group.alb.id]

  # Access logs
  access_logs = {
    bucket  = aws_s3_bucket.alb_logs.id
    prefix  = "alb-logs"
    enabled = true
  }

  target_groups = [
    {
      name_prefix      = "api-"
      backend_protocol = "HTTP"
      backend_port     = 80
      target_type      = "ip"

      health_check = {
        enabled             = true
        healthy_threshold   = 2
        interval            = 30
        matcher             = "200"
        path                = "/health"
        port                = "traffic-port"
        protocol            = "HTTP"
        timeout             = 5
        unhealthy_threshold = 2
      }
    }
  ]

  https_listeners = [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = aws_acm_certificate_validation.main.certificate_arn
      target_group_index = 0

      action_type = "forward"
    }
  ]

  http_tcp_listeners = [
    {
      port        = 80
      protocol    = "HTTP"
      action_type = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  ]

  tags = local.common_tags
}

# S3 Buckets for various purposes
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${local.cluster_name}-alb-logs-${random_string.bucket_suffix.result}"
  force_destroy = var.environment != "production"

  tags = local.common_tags
}

resource "aws_s3_bucket" "terraform_state" {
  bucket        = "${local.cluster_name}-terraform-state-${random_string.bucket_suffix.result}"
  force_destroy = var.environment != "production"

  tags = local.common_tags
}

resource "aws_s3_bucket" "market_data_backup" {
  bucket        = "${local.cluster_name}-market-data-backup-${random_string.bucket_suffix.result}"
  force_destroy = var.environment != "production"

  tags = local.common_tags
}

# Random string for bucket naming
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Random password for Redis
resource "random_password" "redis_auth" {
  length  = 32
  special = true
}

# IAM Roles and Policies
resource "aws_iam_role" "market_data_agent_role" {
  name = "${local.cluster_name}-market-data-agent-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:market-data:market-data-agent"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "market_data_agent_policy" {
  name        = "${local.cluster_name}-market-data-agent-policy"
  description = "IAM policy for market data agent"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.market_data_backup.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.market_data_backup.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.market_data_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "market_data_agent_policy" {
  role       = aws_iam_role.market_data_agent_role.name
  policy_arn = aws_iam_policy.market_data_agent_policy.arn
}

# Secrets Manager for application secrets
resource "aws_secretsmanager_secret" "market_data_secrets" {
  name                    = "${local.cluster_name}-market-data-secrets"
  description             = "Secrets for market data agent"
  recovery_window_in_days = var.environment == "production" ? 30 : 0

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "market_data_secrets" {
  secret_id = aws_secretsmanager_secret.market_data_secrets.id
  secret_string = jsonencode({
    database_url = "postgresql://${var.database_username}:${module.rds.db_instance_password}@${module.rds.db_instance_endpoint}/${var.database_name}"
    redis_url    = "redis://:${random_password.redis_auth.result}@${module.redis.cluster_cache_nodes[0].address}:${module.redis.cluster_cache_nodes[0].port}"
    api_keys = {
      yfinance_key      = var.yfinance_api_key
      alpha_vantage_key = var.alpha_vantage_api_key
    }
  })
}

# Route53 for DNS management
resource "aws_route53_zone" "main" {
  count = var.domain_name != "" ? 1 : 0
  name  = var.domain_name

  tags = local.common_tags
}

resource "aws_route53_record" "main" {
  count   = var.domain_name != "" ? 1 : 0
  zone_id = aws_route53_zone.main[0].zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = module.alb.lb_dns_name
    zone_id                = module.alb.lb_zone_id
    evaluate_target_health = true
  }
}

# ACM Certificate
resource "aws_acm_certificate" "main" {
  count           = var.domain_name != "" ? 1 : 0
  domain_name     = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}"
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}

resource "aws_route53_record" "main_validation" {
  for_each = var.domain_name != "" ? {
    for dvo in aws_acm_certificate.main[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main[0].zone_id
}

resource "aws_acm_certificate_validation" "main" {
  count           = var.domain_name != "" ? 1 : 0
  certificate_arn = aws_acm_certificate.main[0].arn
  validation_record_fqdns = [for record in aws_route53_record.main_validation : record.fqdn]
}