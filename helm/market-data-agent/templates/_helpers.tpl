{{/*
Expand the name of the chart.
*/}}
{{- define "market-data-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "market-data-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "market-data-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "market-data-agent.labels" -}}
helm.sh/chart: {{ include "market-data-agent.chart" . }}
{{ include "market-data-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: market-data-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "market-data-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "market-data-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "market-data-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "market-data-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "market-data-agent.image" -}}
{{- $registryName := .Values.image.registry -}}
{{- $repositoryName := .Values.image.repository -}}
{{- $tag := .Values.image.tag | toString -}}
{{- if .Values.global.imageRegistry }}
    {{- $registryName = .Values.global.imageRegistry -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified postgresql name.
*/}}
{{- define "market-data-agent.postgresql.fullname" -}}
{{- include "common.names.dependency.fullname" (dict "chartName" "postgresql" "chartValues" .Values.postgresql "context" $) -}}
{{- end }}

{{/*
Get the postgresql secret name.
*/}}
{{- define "market-data-agent.postgresql.secretName" -}}
{{- if .Values.postgresql.auth.existingSecret }}
    {{- .Values.postgresql.auth.existingSecret -}}
{{- else }}
    {{- include "market-data-agent.postgresql.fullname" . -}}
{{- end }}
{{- end }}

{{/*
Get the postgresql secret key.
*/}}
{{- define "market-data-agent.postgresql.secretPasswordKey" -}}
{{- if .Values.postgresql.auth.existingSecret }}
    {{- .Values.postgresql.auth.secretKeys.adminPasswordKey | default "postgres-password" -}}
{{- else }}
    {{- "postgres-password" -}}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified redis name.
*/}}
{{- define "market-data-agent.redis.fullname" -}}
{{- include "common.names.dependency.fullname" (dict "chartName" "redis" "chartValues" .Values.redis "context" $) -}}
{{- end }}

{{/*
Get the redis secret name.
*/}}
{{- define "market-data-agent.redis.secretName" -}}
{{- if .Values.redis.auth.existingSecret }}
    {{- .Values.redis.auth.existingSecret -}}
{{- else }}
    {{- include "market-data-agent.redis.fullname" . -}}
{{- end }}
{{- end }}

{{/*
Get the redis secret key.
*/}}
{{- define "market-data-agent.redis.secretPasswordKey" -}}
{{- if .Values.redis.auth.existingSecret }}
    {{- .Values.redis.auth.secretKeys.redis-password | default "redis-password" -}}
{{- else }}
    {{- "redis-password" -}}
{{- end }}
{{- end }}

{{/*
Create postgresql connection URL
*/}}
{{- define "market-data-agent.postgresql.url" -}}
{{- $host := include "market-data-agent.postgresql.fullname" . -}}
{{- $port := .Values.postgresql.primary.service.ports.postgresql | default 5432 -}}
{{- $database := .Values.postgresql.auth.database -}}
{{- $username := .Values.postgresql.auth.username -}}
{{- printf "postgresql://%s:$(POSTGRES_PASSWORD)@%s:%d/%s" $username $host $port $database -}}
{{- end }}

{{/*
Create redis connection URL
*/}}
{{- define "market-data-agent.redis.url" -}}
{{- $host := include "market-data-agent.redis.fullname" . -}}
{{- $port := .Values.redis.master.service.ports.redis | default 6379 -}}
{{- if .Values.redis.auth.enabled -}}
{{- printf "redis://:$(REDIS_PASSWORD)@%s:%d/0" $host $port -}}
{{- else -}}
{{- printf "redis://%s:%d/0" $host $port -}}
{{- end -}}
{{- end }}