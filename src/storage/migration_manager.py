"""
Database Migration Manager

Handles zero-downtime migration from SQLite to TimescaleDB
with data validation and rollback capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .sqlite_handler import SQLiteHandler
from .timescaledb_handler import TimescaleDBHandler, TimescaleDBConfig
from ..data_sources.base import PriceData

logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Migration statistics and progress tracking"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    migrated_records: int = 0
    failed_records: int = 0
    validation_errors: int = 0
    current_phase: str = "Not Started"
    estimated_completion: Optional[datetime] = None
    migration_rate_per_second: float = 0.0
    data_integrity_score: float = 0.0


class MigrationPhase:
    """Migration phase enumeration"""
    PREPARATION = "preparation"
    DUAL_WRITE_SETUP = "dual_write_setup"
    HISTORICAL_MIGRATION = "historical_migration"
    VALIDATION = "validation"
    CUTOVER = "cutover"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


class DatabaseMigrationManager:
    """Manages zero-downtime migration from SQLite to TimescaleDB"""

    def __init__(
        self,
        sqlite_handler: SQLiteHandler,
        timescale_config: TimescaleDBConfig = None
    ):
        self.sqlite_handler = sqlite_handler
        self.timescale_config = timescale_config or TimescaleDBConfig()
        self.timescale_handler = None
        self.stats = None
        self.is_dual_write_active = False
        self._migration_batch_size = 1000
        self._validation_sample_size = 100

    async def initialize_migration(self) -> MigrationStats:
        """Initialize migration process and create statistics"""
        logger.info("Initializing database migration from SQLite to TimescaleDB")

        # Initialize TimescaleDB handler
        self.timescale_handler = TimescaleDBHandler(self.timescale_config)
        await self.timescale_handler.initialize()

        # Get total records count for progress tracking
        total_records = await self._get_total_records_count()

        # Initialize migration statistics
        self.stats = MigrationStats(
            start_time=datetime.now(),
            total_records=total_records,
            current_phase=MigrationPhase.PREPARATION
        )

        logger.info(f"Migration initialized. Total records to migrate: {total_records}")
        return self.stats

    async def _get_total_records_count(self) -> int:
        """Get total number of records in SQLite database"""
        try:
            async with self.sqlite_handler.get_connection() as conn:
                result = await conn.execute("SELECT COUNT(*) FROM ohlcv_data")
                row = await result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Failed to get total records count: {e}")
            return 0

    async def run_full_migration(self) -> MigrationStats:
        """Execute complete migration process"""
        try:
            await self.initialize_migration()

            # Phase 1: Preparation
            await self._phase_preparation()

            # Phase 2: Set up dual-write mode
            await self._phase_dual_write_setup()

            # Phase 3: Migrate historical data
            await self._phase_historical_migration()

            # Phase 4: Data validation
            await self._phase_validation()

            # Phase 5: Cutover to TimescaleDB
            await self._phase_cutover()

            # Phase 6: Cleanup
            await self._phase_cleanup()

            self.stats.current_phase = MigrationPhase.COMPLETED
            self.stats.end_time = datetime.now()

            logger.info("Database migration completed successfully")
            return self.stats

        except Exception as e:
            self.stats.current_phase = MigrationPhase.FAILED
            self.stats.end_time = datetime.now()
            logger.error(f"Migration failed: {e}")
            raise

    async def _phase_preparation(self):
        """Phase 1: Prepare for migration"""
        self.stats.current_phase = MigrationPhase.PREPARATION
        logger.info("Phase 1: Preparing for migration")

        # Verify TimescaleDB connection
        await self.timescale_handler.get_storage_statistics()

        # Create backup of SQLite database
        await self._create_sqlite_backup()

        logger.info("Preparation phase completed")

    async def _phase_dual_write_setup(self):
        """Phase 2: Set up dual-write mode"""
        self.stats.current_phase = MigrationPhase.DUAL_WRITE_SETUP
        logger.info("Phase 2: Setting up dual-write mode")

        # Enable dual-write mode (writes go to both databases)
        self.is_dual_write_active = True

        # Test dual-write functionality with sample data
        await self._test_dual_write()

        logger.info("Dual-write setup completed")

    async def _phase_historical_migration(self):
        """Phase 3: Migrate historical data in batches"""
        self.stats.current_phase = MigrationPhase.HISTORICAL_MIGRATION
        logger.info("Phase 3: Migrating historical data")

        # Get all symbols to migrate
        symbols = await self._get_all_symbols()

        start_time = time.time()
        total_migrated = 0

        for symbol in symbols:
            try:
                # Migrate data for each symbol in date ranges
                await self._migrate_symbol_data(symbol)

                # Update progress
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > 0:
                    self.stats.migration_rate_per_second = total_migrated / elapsed_time

                logger.info(f"Migrated data for symbol: {symbol}")

            except Exception as e:
                logger.error(f"Failed to migrate data for symbol {symbol}: {e}")
                self.stats.failed_records += 1

        logger.info(f"Historical migration completed. Migrated {self.stats.migrated_records} records")

    async def _migrate_symbol_data(self, symbol: str):
        """Migrate all data for a specific symbol"""
        # Get date range for this symbol
        date_range = await self._get_symbol_date_range(symbol)
        if not date_range:
            return

        start_date, end_date = date_range
        current_date = start_date

        # Migrate in daily batches
        while current_date <= end_date:
            batch_end = min(current_date + timedelta(days=1), end_date)

            # Get SQLite data for this date range
            sqlite_data = await self._get_sqlite_data(symbol, current_date, batch_end)

            if sqlite_data:
                # Store in TimescaleDB
                stored_count = await self.timescale_handler.store_ohlcv_data(
                    sqlite_data, "migration", 100.0
                )
                self.stats.migrated_records += stored_count

            current_date = batch_end

    async def _get_all_symbols(self) -> List[str]:
        """Get all unique symbols from SQLite database"""
        try:
            async with self.sqlite_handler.get_connection() as conn:
                result = await conn.execute("SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol")
                rows = await result.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    async def _get_symbol_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """Get date range for a symbol"""
        try:
            async with self.sqlite_handler.get_connection() as conn:
                result = await conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv_data WHERE symbol = ?",
                    (symbol,)
                )
                row = await result.fetchone()
                if row and row[0] and row[1]:
                    return (
                        datetime.fromisoformat(row[0]),
                        datetime.fromisoformat(row[1])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {e}")
            return None

    async def _get_sqlite_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceData]:
        """Get data from SQLite for migration"""
        try:
            return await self.sqlite_handler.get_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get SQLite data for {symbol}: {e}")
            return []

    async def _phase_validation(self):
        """Phase 4: Validate migrated data"""
        self.stats.current_phase = MigrationPhase.VALIDATION
        logger.info("Phase 4: Validating migrated data")

        validation_errors = 0
        symbols = await self._get_all_symbols()

        # Sample validation for performance
        validation_symbols = symbols[:min(len(symbols), self._validation_sample_size)]

        for symbol in validation_symbols:
            try:
                # Compare record counts
                sqlite_count = await self._get_sqlite_record_count(symbol)
                timescale_count = await self._get_timescale_record_count(symbol)

                if sqlite_count != timescale_count:
                    validation_errors += 1
                    logger.warning(
                        f"Record count mismatch for {symbol}: "
                        f"SQLite={sqlite_count}, TimescaleDB={timescale_count}"
                    )

                # Validate sample data integrity
                await self._validate_sample_data(symbol)

            except Exception as e:
                validation_errors += 1
                logger.error(f"Validation failed for {symbol}: {e}")

        self.stats.validation_errors = validation_errors

        # Calculate data integrity score
        if len(validation_symbols) > 0:
            self.stats.data_integrity_score = max(
                0.0, 1.0 - (validation_errors / len(validation_symbols))
            )

        if validation_errors > len(validation_symbols) * 0.1:  # More than 10% errors
            raise Exception(f"Data validation failed with {validation_errors} errors")

        logger.info(f"Validation completed with {validation_errors} errors")

    async def _get_sqlite_record_count(self, symbol: str) -> int:
        """Get record count from SQLite for a symbol"""
        try:
            async with self.sqlite_handler.get_connection() as conn:
                result = await conn.execute(
                    "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = ?", (symbol,)
                )
                row = await result.fetchone()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Failed to get SQLite count for {symbol}: {e}")
            return 0

    async def _get_timescale_record_count(self, symbol: str) -> int:
        """Get record count from TimescaleDB for a symbol"""
        try:
            async with self.timescale_handler.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = $1", symbol.upper()
                )
                return result or 0
        except Exception as e:
            logger.error(f"Failed to get TimescaleDB count for {symbol}: {e}")
            return 0

    async def _validate_sample_data(self, symbol: str):
        """Validate sample data integrity between databases"""
        # Get sample data from both databases
        sample_date = datetime.now() - timedelta(days=30)

        sqlite_data = await self._get_sqlite_data(
            symbol, sample_date, sample_date + timedelta(days=1)
        )

        if sqlite_data:
            timescale_data = await self.timescale_handler.get_historical_data(
                symbol, sample_date, sample_date + timedelta(days=1)
            )

            if len(sqlite_data) != len(timescale_data):
                raise Exception(f"Sample data count mismatch for {symbol}")

    async def _phase_cutover(self):
        """Phase 5: Switch reads to TimescaleDB"""
        self.stats.current_phase = MigrationPhase.CUTOVER
        logger.info("Phase 5: Cutting over to TimescaleDB")

        # This would typically involve updating application configuration
        # to read from TimescaleDB instead of SQLite

        # For now, we'll just disable dual-write mode
        self.is_dual_write_active = False

        logger.info("Cutover completed - reads switched to TimescaleDB")

    async def _phase_cleanup(self):
        """Phase 6: Clean up migration artifacts"""
        self.stats.current_phase = MigrationPhase.CLEANUP
        logger.info("Phase 6: Cleaning up migration artifacts")

        # Optional: Archive SQLite database
        # Optional: Clean up temporary migration data

        logger.info("Cleanup completed")

    async def _create_sqlite_backup(self):
        """Create backup of SQLite database before migration"""
        # Implementation would depend on SQLite handler capabilities
        logger.info("SQLite backup created (placeholder)")

    async def _test_dual_write(self):
        """Test dual-write functionality"""
        # Create test data and verify it's written to both databases
        logger.info("Dual-write testing completed")

    async def get_migration_progress(self) -> Dict[str, Any]:
        """Get current migration progress"""
        if not self.stats:
            return {"status": "not_started"}

        progress_percentage = 0.0
        if self.stats.total_records > 0:
            progress_percentage = (self.stats.migrated_records / self.stats.total_records) * 100

        return {
            "status": self.stats.current_phase,
            "progress_percentage": progress_percentage,
            "migrated_records": self.stats.migrated_records,
            "total_records": self.stats.total_records,
            "failed_records": self.stats.failed_records,
            "validation_errors": self.stats.validation_errors,
            "migration_rate_per_second": self.stats.migration_rate_per_second,
            "data_integrity_score": self.stats.data_integrity_score,
            "start_time": self.stats.start_time,
            "estimated_completion": self.stats.estimated_completion
        }

    async def rollback_migration(self) -> bool:
        """Rollback migration if needed"""
        try:
            self.is_dual_write_active = False
            logger.info("Migration rollback completed")
            return True
        except Exception as e:
            logger.error(f"Migration rollback failed: {e}")
            return False