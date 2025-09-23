#!/bin/bash
# Infrastructure Validation Script for Market Data Agent
# Phase 4 Step 3: Configuration Management & Environment Automation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TERRAFORM_DIR="${PROJECT_ROOT}/terraform"
ANSIBLE_DIR="${PROJECT_ROOT}/ansible"

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

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Validate infrastructure configuration for Market Data Agent

OPTIONS:
    -e, --environment ENV    Environment to validate (dev, staging, prod)
    -t, --terraform         Validate Terraform configuration only
    -a, --ansible           Validate Ansible configuration only
    -k, --kubernetes        Validate Kubernetes configuration only
    -v, --vault             Validate Vault configuration only
    -h, --help              Show this help message

EXAMPLES:
    $0 -e dev                        # Validate all configurations for dev
    $0 -e prod -t                    # Validate only Terraform for prod
    $0 -e staging -a -k              # Validate Ansible and Kubernetes for staging
EOF
}

# Parse command line arguments
ENVIRONMENT=""
VALIDATE_TERRAFORM=false
VALIDATE_ANSIBLE=false
VALIDATE_KUBERNETES=false
VALIDATE_VAULT=false
VALIDATE_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--terraform)
            VALIDATE_TERRAFORM=true
            VALIDATE_ALL=false
            shift
            ;;
        -a|--ansible)
            VALIDATE_ANSIBLE=true
            VALIDATE_ALL=false
            shift
            ;;
        -k|--kubernetes)
            VALIDATE_KUBERNETES=true
            VALIDATE_ALL=false
            shift
            ;;
        -v|--vault)
            VALIDATE_VAULT=true
            VALIDATE_ALL=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "${ENVIRONMENT}" ]]; then
    log_error "Environment is required. Use -e or --environment"
    usage
    exit 1
fi

# Validate environment value
if [[ ! "${ENVIRONMENT}" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Environment must be one of: dev, staging, prod"
    exit 1
fi

# Set validation flags if VALIDATE_ALL is true
if [[ "${VALIDATE_ALL}" == "true" ]]; then
    VALIDATE_TERRAFORM=true
    VALIDATE_ANSIBLE=true
    VALIDATE_KUBERNETES=true
    VALIDATE_VAULT=true
fi

# Global variables for tracking results
VALIDATION_ERRORS=0
VALIDATION_WARNINGS=0

# Function to check required tools
check_prerequisites() {
    log_info "Checking prerequisites..."

    local tools=(
        "terraform:Terraform"
        "ansible:Ansible"
        "kubectl:Kubernetes CLI"
        "helm:Helm"
        "vault:Vault CLI"
        "jq:jq JSON processor"
        "yq:yq YAML processor"
    )

    for tool_info in "${tools[@]}"; do
        IFS=':' read -r tool name <<< "${tool_info}"
        if ! command -v "${tool}" &> /dev/null; then
            log_warn "${name} (${tool}) is not installed or not in PATH"
            ((VALIDATION_WARNINGS++))
        else
            log_success "${name} found: $(command -v "${tool}")"
        fi
    done
}

# Validate Terraform configuration
validate_terraform() {
    log_info "Validating Terraform configuration for ${ENVIRONMENT}..."

    cd "${TERRAFORM_DIR}"

    # Check if tfvars file exists
    local tfvars_file="environments/${ENVIRONMENT}.tfvars"
    if [[ ! -f "${tfvars_file}" ]]; then
        log_error "Terraform variables file not found: ${tfvars_file}"
        ((VALIDATION_ERRORS++))
        return 1
    fi

    # Validate Terraform syntax
    log_info "Checking Terraform syntax..."
    if terraform validate; then
        log_success "Terraform syntax validation passed"
    else
        log_error "Terraform syntax validation failed"
        ((VALIDATION_ERRORS++))
    fi

    # Validate Terraform formatting
    log_info "Checking Terraform formatting..."
    if terraform fmt -check=true -diff=true; then
        log_success "Terraform formatting is correct"
    else
        log_warn "Terraform files are not properly formatted"
        ((VALIDATION_WARNINGS++))
    fi

    # Initialize Terraform (dry run)
    log_info "Initializing Terraform..."
    if terraform init -backend=false; then
        log_success "Terraform initialization successful"
    else
        log_error "Terraform initialization failed"
        ((VALIDATION_ERRORS++))
    fi

    # Plan Terraform (dry run)
    log_info "Creating Terraform plan..."
    if terraform plan -var-file="${tfvars_file}" -out="/tmp/terraform-${ENVIRONMENT}.plan" &> /dev/null; then
        log_success "Terraform plan created successfully"

        # Analyze plan for potential issues
        terraform show -json "/tmp/terraform-${ENVIRONMENT}.plan" > "/tmp/terraform-${ENVIRONMENT}.json"

        # Check for destructive actions
        local destructive_actions=$(jq -r '.resource_changes[] | select(.change.actions[] | contains("delete")) | .address' "/tmp/terraform-${ENVIRONMENT}.json" 2>/dev/null || echo "")
        if [[ -n "${destructive_actions}" ]]; then
            log_warn "Plan contains destructive actions for: ${destructive_actions}"
            ((VALIDATION_WARNINGS++))
        fi

        # Check resource counts
        local resource_count=$(jq -r '.resource_changes | length' "/tmp/terraform-${ENVIRONMENT}.json" 2>/dev/null || echo "0")
        log_info "Plan affects ${resource_count} resources"

    else
        log_error "Terraform plan failed"
        ((VALIDATION_ERRORS++))
    fi

    # Validate security configurations
    log_info "Checking security configurations..."

    # Check for hardcoded secrets
    if grep -r "password\|secret\|key" --include="*.tf" --include="*.tfvars" . | grep -v "var\." | grep -v "#"; then
        log_error "Potential hardcoded secrets found in Terraform files"
        ((VALIDATION_ERRORS++))
    else
        log_success "No hardcoded secrets found"
    fi

    # Check for public access
    if grep -r "0.0.0.0/0" --include="*.tf" . && [[ "${ENVIRONMENT}" == "prod" ]]; then
        log_warn "Public access (0.0.0.0/0) found in production configuration"
        ((VALIDATION_WARNINGS++))
    fi

    cd - > /dev/null
}

# Validate Ansible configuration
validate_ansible() {
    log_info "Validating Ansible configuration for ${ENVIRONMENT}..."

    cd "${ANSIBLE_DIR}"

    # Check inventory file
    local inventory_file="inventory/hosts.yml"
    if [[ ! -f "${inventory_file}" ]]; then
        log_error "Ansible inventory file not found: ${inventory_file}"
        ((VALIDATION_ERRORS++))
        return 1
    fi

    # Validate YAML syntax
    log_info "Checking YAML syntax..."
    if find . -name "*.yml" -o -name "*.yaml" | xargs -I {} sh -c 'echo "Checking {}" && yq eval . {} > /dev/null'; then
        log_success "All YAML files have valid syntax"
    else
        log_error "YAML syntax errors found"
        ((VALIDATION_ERRORS++))
    fi

    # Check Ansible syntax
    log_info "Checking Ansible playbook syntax..."
    if ansible-playbook playbooks/site.yml --syntax-check; then
        log_success "Ansible syntax validation passed"
    else
        log_error "Ansible syntax validation failed"
        ((VALIDATION_ERRORS++))
    fi

    # Validate inventory
    log_info "Validating Ansible inventory..."
    if ansible-inventory --list -i "${inventory_file}" > /dev/null; then
        log_success "Ansible inventory is valid"
    else
        log_error "Ansible inventory validation failed"
        ((VALIDATION_ERRORS++))
    fi

    # Check for environment-specific hosts
    log_info "Checking environment-specific configuration..."
    if ansible-inventory --list -i "${inventory_file}" | jq -r ".${ENVIRONMENT}" | grep -q "hosts"; then
        log_success "Environment ${ENVIRONMENT} found in inventory"
    else
        log_warn "Environment ${ENVIRONMENT} not found in inventory"
        ((VALIDATION_WARNINGS++))
    fi

    # Validate roles
    log_info "Checking Ansible roles..."
    for role_dir in roles/*/; do
        if [[ -d "${role_dir}" ]]; then
            role_name=$(basename "${role_dir}")
            if [[ -f "${role_dir}/tasks/main.yml" ]]; then
                log_success "Role ${role_name} has main tasks file"
            else
                log_warn "Role ${role_name} missing main tasks file"
                ((VALIDATION_WARNINGS++))
            fi
        fi
    done

    cd - > /dev/null
}

# Validate Kubernetes configuration
validate_kubernetes() {
    log_info "Validating Kubernetes configuration for ${ENVIRONMENT}..."

    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_warn "kubectl is not configured or cluster is not accessible"
        ((VALIDATION_WARNINGS++))
        return 1
    fi

    # Validate YAML manifests
    log_info "Checking Kubernetes manifest syntax..."
    find "${PROJECT_ROOT}" -name "*.yaml" -path "*/k8s/*" -o -name "*.yml" -path "*/k8s/*" | while read -r manifest; do
        if kubectl apply --dry-run=client -f "${manifest}" &> /dev/null; then
            log_success "Valid manifest: $(basename "${manifest}")"
        else
            log_error "Invalid manifest: ${manifest}"
            ((VALIDATION_ERRORS++))
        fi
    done

    # Check Helm charts
    log_info "Validating Helm charts..."
    find "${PROJECT_ROOT}" -name "Chart.yaml" | while read -r chart_file; do
        chart_dir=$(dirname "${chart_file}")
        chart_name=$(basename "${chart_dir}")

        if helm lint "${chart_dir}"; then
            log_success "Helm chart ${chart_name} validation passed"
        else
            log_error "Helm chart ${chart_name} validation failed"
            ((VALIDATION_ERRORS++))
        fi

        # Template the chart
        if helm template "${chart_name}" "${chart_dir}" > /dev/null; then
            log_success "Helm chart ${chart_name} templating successful"
        else
            log_error "Helm chart ${chart_name} templating failed"
            ((VALIDATION_ERRORS++))
        fi
    done

    # Check resource quotas and limits
    log_info "Checking resource configurations..."
    if kubectl get resourcequota --all-namespaces &> /dev/null; then
        log_success "Resource quotas are configured"
    else
        log_warn "No resource quotas found"
        ((VALIDATION_WARNINGS++))
    fi

    # Check security policies
    log_info "Checking security policies..."
    if kubectl get podsecuritypolicy &> /dev/null; then
        log_success "Pod security policies are configured"
    else
        log_warn "No pod security policies found"
        ((VALIDATION_WARNINGS++))
    fi
}

# Validate Vault configuration
validate_vault() {
    log_info "Validating Vault configuration for ${ENVIRONMENT}..."

    local vault_config="${PROJECT_ROOT}/vault/vault-config.hcl"

    # Check Vault configuration file
    if [[ ! -f "${vault_config}" ]]; then
        log_error "Vault configuration file not found: ${vault_config}"
        ((VALIDATION_ERRORS++))
        return 1
    fi

    # Validate HCL syntax
    log_info "Checking Vault HCL syntax..."
    if vault validate "${vault_config}"; then
        log_success "Vault configuration syntax is valid"
    else
        log_error "Vault configuration syntax is invalid"
        ((VALIDATION_ERRORS++))
    fi

    # Check Vault Kubernetes manifests
    log_info "Checking Vault Kubernetes manifests..."
    find "${PROJECT_ROOT}/vault/kubernetes" -name "*.yaml" -o -name "*.yml" | while read -r manifest; do
        if kubectl apply --dry-run=client -f "${manifest}" &> /dev/null; then
            log_success "Valid Vault manifest: $(basename "${manifest}")"
        else
            log_error "Invalid Vault manifest: ${manifest}"
            ((VALIDATION_ERRORS++))
        fi
    done

    # Check Vault Helm values
    local vault_values="${PROJECT_ROOT}/vault/helm/values.yaml"
    if [[ -f "${vault_values}" ]]; then
        if yq eval . "${vault_values}" > /dev/null; then
            log_success "Vault Helm values file is valid YAML"
        else
            log_error "Vault Helm values file has invalid YAML"
            ((VALIDATION_ERRORS++))
        fi
    fi

    # Check for security configurations
    log_info "Checking Vault security configurations..."

    # Check for TLS configuration
    if grep -q "tls_disable.*false" "${vault_config}"; then
        log_success "TLS is enabled in Vault configuration"
    else
        log_warn "TLS may not be properly configured in Vault"
        ((VALIDATION_WARNINGS++))
    fi

    # Check for seal configuration
    if grep -q "seal.*awskms" "${vault_config}"; then
        log_success "AWS KMS seal is configured"
    else
        log_warn "AWS KMS seal is not configured (acceptable for dev/staging)"
        if [[ "${ENVIRONMENT}" == "prod" ]]; then
            ((VALIDATION_WARNINGS++))
        fi
    fi
}

# Generate validation report
generate_report() {
    log_info "Generating validation report..."

    local report_file="/tmp/infrastructure-validation-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"

    cat > "${report_file}" << EOF
{
    "environment": "${ENVIRONMENT}",
    "validation_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "validation_components": {
        "terraform": ${VALIDATE_TERRAFORM},
        "ansible": ${VALIDATE_ANSIBLE},
        "kubernetes": ${VALIDATE_KUBERNETES},
        "vault": ${VALIDATE_VAULT}
    },
    "results": {
        "errors": ${VALIDATION_ERRORS},
        "warnings": ${VALIDATION_WARNINGS},
        "status": "$([[ ${VALIDATION_ERRORS} -eq 0 ]] && echo "PASS" || echo "FAIL")"
    },
    "recommendations": [
EOF

    # Add recommendations based on findings
    if [[ ${VALIDATION_ERRORS} -gt 0 ]]; then
        echo '        "Fix all validation errors before deployment",' >> "${report_file}"
    fi

    if [[ ${VALIDATION_WARNINGS} -gt 0 ]]; then
        echo '        "Review and address validation warnings",' >> "${report_file}"
    fi

    if [[ "${ENVIRONMENT}" == "prod" ]] && [[ ${VALIDATION_WARNINGS} -gt 0 ]]; then
        echo '        "Production environment should have minimal warnings",' >> "${report_file}"
    fi

    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "${report_file}"
    echo '    ]' >> "${report_file}"
    echo '}' >> "${report_file}"

    log_info "Validation report saved to: ${report_file}"

    # Display summary
    echo
    log_info "=== VALIDATION SUMMARY ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Errors: ${VALIDATION_ERRORS}"
    log_info "Warnings: ${VALIDATION_WARNINGS}"

    if [[ ${VALIDATION_ERRORS} -eq 0 ]]; then
        log_success "✅ Validation PASSED"
        return 0
    else
        log_error "❌ Validation FAILED"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting infrastructure validation for ${ENVIRONMENT} environment..."
    echo

    check_prerequisites
    echo

    if [[ "${VALIDATE_TERRAFORM}" == "true" ]]; then
        validate_terraform
        echo
    fi

    if [[ "${VALIDATE_ANSIBLE}" == "true" ]]; then
        validate_ansible
        echo
    fi

    if [[ "${VALIDATE_KUBERNETES}" == "true" ]]; then
        validate_kubernetes
        echo
    fi

    if [[ "${VALIDATE_VAULT}" == "true" ]]; then
        validate_vault
        echo
    fi

    generate_report
}

# Execute main function
main "$@"