# Security Groups for Market Data Agent Infrastructure
# Phase 4 Step 3: Configuration Management & Environment Automation

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${local.cluster_name}-alb-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for Application Load Balancer"

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-alb-sg"
    Type = "alb"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${local.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for RDS PostgreSQL database"

  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    description     = "PostgreSQL from EKS cluster"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  # Allow access from specific CIDR blocks for database administration
  dynamic "ingress" {
    for_each = var.environment == "dev" ? [1] : []
    content {
      description = "PostgreSQL from admin networks"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidr_blocks
    }
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
    Type = "database"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name_prefix = "${local.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for Redis ElastiCache cluster"

  ingress {
    description     = "Redis from EKS nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    description     = "Redis from EKS cluster"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  # Allow access from specific CIDR blocks for cache administration
  dynamic "ingress" {
    for_each = var.environment == "dev" ? [1] : []
    content {
      description = "Redis from admin networks"
      from_port   = 6379
      to_port     = 6379
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidr_blocks
    }
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-sg"
    Type = "cache"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Bastion Host Security Group (optional, for troubleshooting)
resource "aws_security_group" "bastion" {
  count = var.environment == "dev" ? 1 : 0

  name_prefix = "${local.cluster_name}-bastion-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for bastion host"

  ingress {
    description = "SSH from admin networks"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-bastion-sg"
    Type = "bastion"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Monitoring Security Group
resource "aws_security_group" "monitoring" {
  name_prefix = "${local.cluster_name}-monitoring-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for monitoring services"

  # Prometheus
  ingress {
    description     = "Prometheus from EKS nodes"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  # Grafana
  ingress {
    description     = "Grafana from ALB"
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # AlertManager
  ingress {
    description     = "AlertManager from EKS nodes"
    from_port       = 9093
    to_port         = 9093
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  # Jaeger
  ingress {
    description     = "Jaeger Query from ALB"
    from_port       = 16686
    to_port         = 16686
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Jaeger Collector gRPC from EKS nodes"
    from_port       = 14250
    to_port         = 14250
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  ingress {
    description     = "Jaeger Collector HTTP from EKS nodes"
    from_port       = 14268
    to_port         = 14268
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  # Elasticsearch
  ingress {
    description     = "Elasticsearch from EKS nodes"
    from_port       = 9200
    to_port         = 9200
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  # Kibana
  ingress {
    description     = "Kibana from ALB"
    from_port       = 5601
    to_port         = 5601
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Logstash
  ingress {
    description     = "Logstash Beats from EKS nodes"
    from_port       = 5044
    to_port         = 5044
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-monitoring-sg"
    Type = "monitoring"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# VPC Endpoints Security Group
resource "aws_security_group" "vpc_endpoints" {
  name_prefix = "${local.cluster_name}-vpc-endpoints-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for VPC endpoints"

  ingress {
    description = "HTTPS from VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [local.vpc_cidr]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-vpc-endpoints-sg"
    Type = "vpc-endpoints"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# WAF Security Group (if WAF is enabled)
resource "aws_security_group" "waf" {
  count = var.enable_waf ? 1 : 0

  name_prefix = "${local.cluster_name}-waf-"
  vpc_id      = module.vpc.vpc_id

  description = "Security group for WAF protected resources"

  ingress {
    description = "HTTP from anywhere (protected by WAF)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS from anywhere (protected by WAF)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-waf-sg"
    Type = "waf"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Additional security group rules for monitoring
resource "aws_security_group_rule" "monitoring_metrics" {
  type                     = "ingress"
  from_port                = 8080
  to_port                  = 8090
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.monitoring.id
  security_group_id        = module.eks.node_security_group_id
  description              = "Allow monitoring to scrape metrics from nodes"
}

# Security group rule for health checks
resource "aws_security_group_rule" "alb_health_check" {
  type                     = "ingress"
  from_port                = 8080
  to_port                  = 8080
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.alb.id
  security_group_id        = module.eks.node_security_group_id
  description              = "Allow ALB health checks"
}

# ElastiCache subnet group
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.cluster_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-subnet-group"
  })
}