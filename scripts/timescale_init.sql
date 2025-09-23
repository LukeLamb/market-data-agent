-- TimescaleDB Initialization Script
-- Phase 3 Step 1: Time-Series Database Schema Setup

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create optimized indexes for performance
-- These will be created after hypertables are set up by the handler

-- Performance tuning settings
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
ALTER SYSTEM SET max_worker_processes = 16;
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- TimescaleDB specific settings
SELECT set_config('timescaledb.max_background_workers', '8', false);

-- Create roles and permissions
DO $$
BEGIN
    -- Create read-only role for analytics
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'market_reader') THEN
        CREATE ROLE market_reader;
    END IF;

    -- Grant permissions
    GRANT CONNECT ON DATABASE market_data TO market_reader;
    GRANT USAGE ON SCHEMA public TO market_reader;

    -- These permissions will be granted after tables are created
    -- GRANT SELECT ON ALL TABLES IN SCHEMA public TO market_reader;

END $$;

-- Log initialization
INSERT INTO pg_stat_statements_info (query)
VALUES ('TimescaleDB initialized for Market Data Agent Phase 3')
ON CONFLICT DO NOTHING;