#!/bin/bash
# Vault Initialization Script for Market Data Agent
# Phase 4 Step 3: Configuration Management & Environment Automation

set -euo pipefail

# Configuration
VAULT_NAMESPACE="vault"
VAULT_SERVICE="vault"
VAULT_PORT="8200"
VAULT_ADDR="https://${VAULT_SERVICE}.${VAULT_NAMESPACE}.svc.cluster.local:${VAULT_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Vault is running
check_vault_status() {
    log_info "Checking Vault status..."

    if ! kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault | grep -q Running; then
        log_error "Vault pods are not running. Please deploy Vault first."
        exit 1
    fi

    log_success "Vault pods are running"
}

# Initialize Vault
initialize_vault() {
    log_info "Initializing Vault..."

    # Check if Vault is already initialized
    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    INIT_STATUS=$(kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -- vault status -format=json | jq -r '.initialized // false')

    if [ "${INIT_STATUS}" = "true" ]; then
        log_warn "Vault is already initialized"
        return 0
    fi

    # Initialize Vault with 5 key shares and threshold of 3
    log_info "Initializing Vault with key shares..."

    INIT_OUTPUT=$(kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -- vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json)

    # Store keys securely (in production, use a secure key management system)
    echo "${INIT_OUTPUT}" > vault-init-keys.json
    chmod 600 vault-init-keys.json

    log_success "Vault initialized successfully"
    log_warn "IMPORTANT: Store the vault-init-keys.json file securely!"

    # Extract unseal keys and root token
    UNSEAL_KEYS=$(echo "${INIT_OUTPUT}" | jq -r '.unseal_keys_b64[]')
    ROOT_TOKEN=$(echo "${INIT_OUTPUT}" | jq -r '.root_token')

    # Unseal Vault
    log_info "Unsealing Vault..."
    local count=0
    for key in ${UNSEAL_KEYS}; do
        if [ ${count} -lt 3 ]; then
            kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -- vault operator unseal ${key}
            ((count++))
        fi
    done

    log_success "Vault unsealed successfully"

    # Set root token for subsequent operations
    export VAULT_TOKEN=${ROOT_TOKEN}
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "export VAULT_TOKEN=${ROOT_TOKEN} && vault auth -method=token"
}

# Configure Kubernetes authentication
configure_kubernetes_auth() {
    log_info "Configuring Kubernetes authentication..."

    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    # Enable Kubernetes auth method
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault auth enable kubernetes || true
    "

    # Get Kubernetes cluster information
    KUBERNETES_HOST=$(kubectl config view --raw --minify --flatten -o jsonpath='{.clusters[].cluster.server}')
    KUBERNETES_CA_CERT=$(kubectl get secret -n vault vault-token-* -o jsonpath='{.data.ca\.crt}' | base64 -d)
    TOKEN_REVIEWER_JWT=$(kubectl get secret -n vault vault-token-* -o jsonpath='{.data.token}' | base64 -d)

    # Configure Kubernetes auth
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault write auth/kubernetes/config \
            token_reviewer_jwt=\"${TOKEN_REVIEWER_JWT}\" \
            kubernetes_host=\"${KUBERNETES_HOST}\" \
            kubernetes_ca_cert=\"${KUBERNETES_CA_CERT}\" \
            issuer=\"https://kubernetes.default.svc.cluster.local\"
    "

    log_success "Kubernetes authentication configured"
}

# Create policies
create_policies() {
    log_info "Creating Vault policies..."

    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    # Market Data Agent policy
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault policy write market-data-agent - <<EOF
# Read secrets for market data agent
path \"secret/data/market-data/*\" {
  capabilities = [\"read\"]
}

# Read database credentials
path \"secret/data/database/*\" {
  capabilities = [\"read\"]
}

# Read API keys
path \"secret/data/api-keys/*\" {
  capabilities = [\"read\"]
}

# Read cache configuration
path \"secret/data/cache/*\" {
  capabilities = [\"read\"]
}

# Allow renewal of own token
path \"auth/token/renew-self\" {
  capabilities = [\"update\"]
}

# Allow looking up own token
path \"auth/token/lookup-self\" {
  capabilities = [\"read\"]
}
EOF
    "

    # Monitoring policy
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault policy write monitoring - <<EOF
# Read monitoring secrets
path \"secret/data/monitoring/*\" {
  capabilities = [\"read\"]
}

# Read alerting credentials
path \"secret/data/alerting/*\" {
  capabilities = [\"read\"]
}
EOF
    "

    log_success "Vault policies created"
}

# Create roles
create_roles() {
    log_info "Creating Vault roles..."

    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    # Market Data Agent role
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault write auth/kubernetes/role/market-data-agent \
            bound_service_account_names=market-data-agent-vault \
            bound_service_account_namespaces=market-data \
            policies=market-data-agent \
            ttl=1h \
            max_ttl=24h
    "

    # Monitoring role
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault write auth/kubernetes/role/monitoring \
            bound_service_account_names=monitoring \
            bound_service_account_namespaces=monitoring \
            policies=monitoring \
            ttl=1h \
            max_ttl=24h
    "

    log_success "Vault roles created"
}

# Enable secret engines
enable_secret_engines() {
    log_info "Enabling secret engines..."

    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    # Enable KV v2 secret engine
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault secrets enable -path=secret kv-v2 || true
    "

    # Enable database secret engine
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault secrets enable database || true
    "

    log_success "Secret engines enabled"
}

# Store initial secrets
store_initial_secrets() {
    log_info "Storing initial secrets..."

    VAULT_POD=$(kubectl get pods -n ${VAULT_NAMESPACE} -l app.kubernetes.io/name=vault --no-headers -o custom-columns=":metadata.name" | head -1)

    # Store API keys (replace with actual values)
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault kv put secret/api-keys/yfinance \
            api_key=\"your-yfinance-api-key\"

        vault kv put secret/api-keys/alpha-vantage \
            api_key=\"your-alpha-vantage-api-key\"
    "

    # Store database configuration
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault kv put secret/database/postgres \
            host=\"postgres.market-data.svc.cluster.local\" \
            port=\"5432\" \
            database=\"marketdata\" \
            username=\"marketdata_user\"
    "

    # Store cache configuration
    kubectl exec -n ${VAULT_NAMESPACE} ${VAULT_POD} -i -- sh -c "
        export VAULT_TOKEN=${VAULT_TOKEN}
        vault kv put secret/cache/redis \
            host=\"redis.market-data.svc.cluster.local\" \
            port=\"6379\"
    "

    log_success "Initial secrets stored"
}

# Main execution
main() {
    log_info "Starting Vault initialization for Market Data Agent..."

    check_vault_status
    initialize_vault
    configure_kubernetes_auth
    create_policies
    create_roles
    enable_secret_engines
    store_initial_secrets

    log_success "Vault initialization completed successfully!"
    log_info "Next steps:"
    log_info "1. Securely store the vault-init-keys.json file"
    log_info "2. Update your application to use Vault for secret management"
    log_info "3. Test secret retrieval from your application"
}

# Check if running directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi