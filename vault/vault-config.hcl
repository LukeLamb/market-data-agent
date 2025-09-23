# HashiCorp Vault Configuration for Market Data Agent
# Phase 4 Step 3: Configuration Management & Environment Automation

# Storage backend configuration
storage "consul" {
  address = "consul.vault.svc.cluster.local:8500"
  path    = "vault/"

  # Consul ACL token for Vault (set via environment variable)
  # CONSUL_HTTP_TOKEN
}

# Alternative: File storage for development/testing
# storage "file" {
#   path = "/vault/data"
# }

# Alternative: AWS S3 storage for production
# storage "s3" {
#   bucket     = "market-data-agent-vault-storage"
#   region     = "us-west-2"
#   kms_key_id = "alias/vault-key"
# }

# Listener configuration
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/vault.crt"
  tls_key_file  = "/vault/tls/vault.key"

  # TLS configuration
  tls_min_version = "tls12"
  tls_cipher_suites = [
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256"
  ]
}

# API address
api_addr = "https://vault.vault.svc.cluster.local:8200"

# Cluster address for Vault Enterprise
cluster_addr = "https://vault.vault.svc.cluster.local:8201"

# UI configuration
ui = true

# Logging configuration
log_level = "INFO"
log_format = "json"

# Disable mlock for containerized environments
disable_mlock = true

# Raw storage endpoint (disable for production)
raw_storage_endpoint = false

# Maximum lease TTL
max_lease_ttl = "8760h"
default_lease_ttl = "168h"

# Plugin directory
plugin_directory = "/vault/plugins"

# Seal configuration (AWS KMS for production)
seal "awskms" {
  region     = "us-west-2"
  kms_key_id = "alias/vault-seal-key"
  endpoint   = "https://kms.us-west-2.amazonaws.com"
}

# Telemetry configuration
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true

  # StatsD configuration
  statsd_address = "localhost:8125"

  # Circonus configuration (optional)
  # circonus_api_token = "your-circonus-token"
  # circonus_api_app = "vault"
  # circonus_api_url = "https://api.circonus.com/v2"
  # circonus_submission_interval = "10s"
}

# Entropy configuration
entropy "seal" {
  mode = "augmentation"
}