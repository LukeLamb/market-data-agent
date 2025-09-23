"""
Bulk Historical Data Importer for Market Data Agent
Handles large-scale historical data imports with parallel processing and progress tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from decimal import Decimal

from ..data_sources.base import PriceData
from ..storage.hybrid_storage_service import HybridStorageService
from ..caching.redis_cache_manager import RedisCacheManager

logger = logging.getLogger(__name__)


class ImportFormat(Enum):
    """Supported import data formats"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    XLSX = "xlsx"
    TSV = "tsv"


class ImportStatus(Enum):
    """Import job status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationLevel(Enum):
    """Data validation levels"""
    NONE = "none"           # No validation
    BASIC = "basic"         # Basic format validation
    STRICT = "strict"       # Comprehensive validation
    CUSTOM = "custom"       # Custom validation rules


@dataclass
class ImportConfig:
    """Configuration for bulk data import operations"""
    # Processing settings
    batch_size: int = 10000
    max_workers: int = mp.cpu_count()
    chunk_size: int = 100000
    parallel_processing: bool = True

    # Data validation
    validation_level: ValidationLevel = ValidationLevel.STRICT
    skip_duplicates: bool = True
    allow_partial_failures: bool = True
    max_error_rate: float = 0.05  # 5% max error rate

    # Performance tuning
    use_bulk_insert: bool = True
    disable_constraints: bool = False
    optimize_indexes: bool = True
    vacuum_after_import: bool = True

    # Progress tracking
    enable_progress_tracking: bool = True
    checkpoint_frequency: int = 50000  # Records between checkpoints
    enable_resumption: bool = True

    # Data transformation
    normalize_symbols: bool = True
    validate_timestamps: bool = True
    require_quality_score: bool = False
    default_quality_score: int = 75


@dataclass
class ImportProgress:
    """Progress tracking for import operations"""
    job_id: str
    total_records: int = 0
    processed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0

    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None

    current_file: Optional[str] = None
    current_batch: int = 0
    total_batches: int = 0

    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def progress_percentage(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100

    @property
    def success_rate(self) -> float:
        if self.processed_records == 0:
            return 100.0
        return (self.successful_records / self.processed_records) * 100

    @property
    def records_per_second(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return 0.0
        return self.processed_records / elapsed

    def estimate_completion(self) -> None:
        """Estimate completion time based on current progress"""
        if self.processed_records == 0 or self.total_records == 0:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining_records = self.total_records - self.processed_records

        if self.processed_records > 0:
            rate = self.processed_records / elapsed
            if rate > 0:
                remaining_seconds = remaining_records / rate
                self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)


@dataclass
class ImportJob:
    """Import job definition and state"""
    job_id: str
    name: str
    files: List[str]
    format: ImportFormat
    config: ImportConfig
    status: ImportStatus = ImportStatus.PENDING
    progress: ImportProgress = field(default_factory=lambda: ImportProgress(""))

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.progress.job_id:
            self.progress.job_id = self.job_id


class DataValidator:
    """Data validation for import operations"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.custom_validators: List[Callable] = []

    def add_custom_validator(self, validator: Callable[[Dict[str, Any]], bool]) -> None:
        """Add custom validation function"""
        self.custom_validators.append(validator)

    def validate_record(self, record: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single record"""
        errors = []

        if self.validation_level == ValidationLevel.NONE:
            return True, errors

        # Basic validation
        if self.validation_level in [ValidationLevel.BASIC, ValidationLevel.STRICT]:
            errors.extend(self._basic_validation(record))

        # Strict validation
        if self.validation_level == ValidationLevel.STRICT:
            errors.extend(self._strict_validation(record))

        # Custom validation
        if self.validation_level == ValidationLevel.CUSTOM:
            errors.extend(self._custom_validation(record))

        return len(errors) == 0, errors

    def _basic_validation(self, record: Dict[str, Any]) -> List[str]:
        """Basic format validation"""
        errors = []

        # Required fields
        required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        return errors

    def _strict_validation(self, record: Dict[str, Any]) -> List[str]:
        """Comprehensive validation"""
        errors = []

        try:
            # Validate symbol
            if 'symbol' in record:
                symbol = str(record['symbol']).strip().upper()
                if not symbol or len(symbol) > 20:
                    errors.append("Invalid symbol format")

            # Validate timestamp
            if 'timestamp' in record:
                if isinstance(record['timestamp'], str):
                    try:
                        datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    except ValueError:
                        errors.append("Invalid timestamp format")

            # Validate prices
            price_fields = ['open', 'high', 'low', 'close']
            prices = {}

            for field in price_fields:
                if field in record:
                    try:
                        price = float(record[field])
                        if price <= 0:
                            errors.append(f"Invalid {field} price: must be positive")
                        prices[field] = price
                    except (ValueError, TypeError):
                        errors.append(f"Invalid {field} price format")

            # Validate price relationships
            if len(prices) == 4:
                if not (prices['low'] <= prices['open'] <= prices['high']):
                    errors.append("Open price outside high-low range")
                if not (prices['low'] <= prices['close'] <= prices['high']):
                    errors.append("Close price outside high-low range")
                if prices['high'] < prices['low']:
                    errors.append("High price less than low price")

            # Validate volume
            if 'volume' in record:
                try:
                    volume = int(record['volume'])
                    if volume < 0:
                        errors.append("Volume cannot be negative")
                except (ValueError, TypeError):
                    errors.append("Invalid volume format")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors

    def _custom_validation(self, record: Dict[str, Any]) -> List[str]:
        """Custom validation using registered validators"""
        errors = []

        for validator in self.custom_validators:
            try:
                if not validator(record):
                    errors.append("Custom validation failed")
            except Exception as e:
                errors.append(f"Custom validation error: {str(e)}")

        return errors


class DataTransformer:
    """Data transformation for import operations"""

    def __init__(self, config: ImportConfig):
        self.config = config

    def transform_record(self, record: Dict[str, Any]) -> PriceData:
        """Transform raw record to PriceData object"""
        try:
            # Normalize symbol
            symbol = str(record['symbol']).strip().upper() if self.config.normalize_symbols else str(record['symbol'])

            # Parse timestamp
            timestamp = record['timestamp']
            if isinstance(timestamp, str):
                # Handle various timestamp formats
                timestamp = timestamp.replace('Z', '+00:00')
                timestamp = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)

            # Convert prices to Decimal
            open_price = Decimal(str(record['open']))
            high_price = Decimal(str(record['high']))
            low_price = Decimal(str(record['low']))
            close_price = Decimal(str(record['close']))

            # Convert volume to int
            volume = int(record['volume'])

            # Handle optional fields
            source = record.get('source', 'bulk_import')
            quality_score = record.get('quality_score', self.config.default_quality_score)

            return PriceData(
                symbol=symbol,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                source=source,
                quality_score=quality_score
            )

        except Exception as e:
            raise ValueError(f"Data transformation failed: {str(e)}")


class FileProcessor:
    """File processing for different formats"""

    @staticmethod
    async def read_csv_file(file_path: str, chunk_size: int = 10000) -> AsyncGenerator[pd.DataFrame, None]:
        """Read CSV file in chunks"""
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    @staticmethod
    async def read_json_file(file_path: str, chunk_size: int = 10000) -> AsyncGenerator[List[Dict], None]:
        """Read JSON file in chunks"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

                if isinstance(data, list):
                    # Process in chunks
                    for i in range(0, len(data), chunk_size):
                        yield data[i:i + chunk_size]
                else:
                    yield [data]
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise

    @staticmethod
    async def read_parquet_file(file_path: str, chunk_size: int = 10000) -> AsyncGenerator[pd.DataFrame, None]:
        """Read Parquet file in chunks"""
        try:
            df = pd.read_parquet(file_path)
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i:i + chunk_size]
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            raise


class BulkDataImporter:
    """Main bulk data importer with parallel processing and progress tracking"""

    def __init__(
        self,
        storage_service: HybridStorageService,
        cache_manager: RedisCacheManager
    ):
        self.storage_service = storage_service
        self.cache_manager = cache_manager

        # Job management
        self.active_jobs: Dict[str, ImportJob] = {}
        self.job_history: List[ImportJob] = []

        # Processing components
        self.validator = DataValidator()
        self.transformer = DataTransformer(ImportConfig())

        # Progress tracking
        self.progress_callbacks: List[Callable[[ImportProgress], None]] = []

        # Resource management
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize the bulk importer"""
        try:
            # Initialize executors
            self.thread_executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
            self.process_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())

            self.is_initialized = True
            logger.info("Bulk data importer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize bulk importer: {e}")
            raise

    async def create_import_job(
        self,
        name: str,
        files: List[str],
        format: ImportFormat,
        config: ImportConfig = None
    ) -> str:
        """Create a new import job"""
        if not self.is_initialized:
            raise RuntimeError("Bulk importer not initialized")

        # Generate job ID
        job_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Create job
        job = ImportJob(
            job_id=job_id,
            name=name,
            files=files,
            format=format,
            config=config or ImportConfig()
        )

        # Initialize progress
        job.progress = ImportProgress(job_id)

        # Estimate total records
        job.progress.total_records = await self._estimate_total_records(files, format)

        self.active_jobs[job_id] = job

        logger.info(f"Created import job {job_id}: {name}")
        return job_id

    async def start_import_job(self, job_id: str) -> bool:
        """Start an import job"""
        if job_id not in self.active_jobs:
            raise ValueError(f"Import job {job_id} not found")

        job = self.active_jobs[job_id]

        if job.status != ImportStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending status")

        try:
            job.status = ImportStatus.RUNNING
            job.started_at = datetime.now()

            # Start import process
            await self._process_import_job(job)

            return True

        except Exception as e:
            job.status = ImportStatus.FAILED
            job.progress.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "type": "job_start_error"
            })
            logger.error(f"Failed to start import job {job_id}: {e}")
            return False

    async def _process_import_job(self, job: ImportJob) -> None:
        """Process import job with parallel processing"""
        try:
            # Setup validator and transformer for this job
            validator = DataValidator(job.config.validation_level)
            transformer = DataTransformer(job.config)

            # Process each file
            for file_path in job.files:
                if job.status != ImportStatus.RUNNING:
                    break

                job.progress.current_file = file_path
                logger.info(f"Processing file: {file_path}")

                await self._process_file(job, file_path, validator, transformer)

                # Update progress
                await self._update_progress(job)

            # Finalize import
            if job.status == ImportStatus.RUNNING:
                await self._finalize_import(job)
                job.status = ImportStatus.COMPLETED
                job.completed_at = datetime.now()

                logger.info(f"Import job {job.job_id} completed successfully")

        except Exception as e:
            job.status = ImportStatus.FAILED
            job.progress.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "type": "processing_error"
            })
            logger.error(f"Import job {job.job_id} failed: {e}")

    async def _process_file(
        self,
        job: ImportJob,
        file_path: str,
        validator: DataValidator,
        transformer: DataTransformer
    ) -> None:
        """Process a single file"""
        try:
            # Determine file reader based on format
            if job.format == ImportFormat.CSV:
                reader = FileProcessor.read_csv_file
            elif job.format == ImportFormat.JSON:
                reader = FileProcessor.read_json_file
            elif job.format == ImportFormat.PARQUET:
                reader = FileProcessor.read_parquet_file
            else:
                raise ValueError(f"Unsupported format: {job.format}")

            # Process file in chunks
            async for chunk in reader(file_path, job.config.chunk_size):
                if job.status != ImportStatus.RUNNING:
                    break

                await self._process_chunk(job, chunk, validator, transformer)

                # Checkpoint if needed
                if (job.progress.processed_records % job.config.checkpoint_frequency == 0 and
                    job.config.enable_progress_tracking):
                    await self._save_checkpoint(job)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    async def _process_chunk(
        self,
        job: ImportJob,
        chunk: Union[pd.DataFrame, List[Dict]],
        validator: DataValidator,
        transformer: DataTransformer
    ) -> None:
        """Process a data chunk with parallel processing"""
        try:
            # Convert chunk to records
            if isinstance(chunk, pd.DataFrame):
                records = chunk.to_dict('records')
            else:
                records = chunk

            # Process records in batches
            batch_size = job.config.batch_size
            for i in range(0, len(records), batch_size):
                if job.status != ImportStatus.RUNNING:
                    break

                batch = records[i:i + batch_size]
                await self._process_batch(job, batch, validator, transformer)

                job.progress.current_batch += 1
                await self._update_progress(job)

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    async def _process_batch(
        self,
        job: ImportJob,
        batch: List[Dict[str, Any]],
        validator: DataValidator,
        transformer: DataTransformer
    ) -> None:
        """Process a batch of records"""
        valid_records = []

        for record in batch:
            job.progress.processed_records += 1

            try:
                # Validate record
                is_valid, errors = validator.validate_record(record)

                if not is_valid:
                    job.progress.failed_records += 1
                    job.progress.errors.append({
                        "record": record,
                        "errors": errors,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Check error rate
                    error_rate = job.progress.failed_records / job.progress.processed_records
                    if error_rate > job.config.max_error_rate and not job.config.allow_partial_failures:
                        raise ValueError(f"Error rate {error_rate:.2%} exceeds maximum {job.config.max_error_rate:.2%}")

                    continue

                # Transform record
                price_data = transformer.transform_record(record)
                valid_records.append(price_data)

            except Exception as e:
                job.progress.failed_records += 1
                job.progress.errors.append({
                    "record": record,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Bulk insert valid records
        if valid_records:
            try:
                await self._bulk_insert_records(valid_records)
                job.progress.successful_records += len(valid_records)
            except Exception as e:
                job.progress.failed_records += len(valid_records)
                logger.error(f"Bulk insert failed: {e}")

    async def _bulk_insert_records(self, records: List[PriceData]) -> None:
        """Bulk insert records into storage"""
        try:
            # Use storage service bulk insert if available
            await self.storage_service.bulk_insert_price_data(records)

        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise

    async def _estimate_total_records(self, files: List[str], format: ImportFormat) -> int:
        """Estimate total records across all files"""
        total = 0

        for file_path in files:
            try:
                if format == ImportFormat.CSV:
                    # Quick count for CSV
                    with open(file_path, 'r') as f:
                        total += sum(1 for line in f) - 1  # Subtract header
                elif format == ImportFormat.JSON:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total += len(data)
                        else:
                            total += 1
                elif format == ImportFormat.PARQUET:
                    df = pd.read_parquet(file_path)
                    total += len(df)

            except Exception as e:
                logger.warning(f"Could not estimate records for {file_path}: {e}")

        return total

    async def _update_progress(self, job: ImportJob) -> None:
        """Update job progress and notify callbacks"""
        job.progress.last_update = datetime.now()
        job.progress.estimate_completion()

        # Notify progress callbacks
        for callback in self.progress_callbacks:
            try:
                await callback(job.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def _save_checkpoint(self, job: ImportJob) -> None:
        """Save job checkpoint for resumption"""
        if not job.config.enable_resumption:
            return

        try:
            checkpoint_data = {
                "job_id": job.job_id,
                "progress": {
                    "processed_records": job.progress.processed_records,
                    "successful_records": job.progress.successful_records,
                    "failed_records": job.progress.failed_records,
                    "current_file": job.progress.current_file,
                    "current_batch": job.progress.current_batch
                },
                "timestamp": datetime.now().isoformat()
            }

            # Save checkpoint to cache
            await self.cache_manager.set(
                f"import_checkpoint:{job.job_id}",
                json.dumps(checkpoint_data),
                ttl=86400  # 24 hours
            )

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def _finalize_import(self, job: ImportJob) -> None:
        """Finalize import with optimization tasks"""
        try:
            if job.config.optimize_indexes:
                # Rebuild indexes for better performance
                logger.info(f"Optimizing indexes for job {job.job_id}")
                # Implementation would depend on storage service

            if job.config.vacuum_after_import:
                # Vacuum database for space reclamation
                logger.info(f"Vacuuming database for job {job.job_id}")
                # Implementation would depend on storage service

            # Clear checkpoint
            if job.config.enable_resumption:
                await self.cache_manager.delete(f"import_checkpoint:{job.job_id}")

        except Exception as e:
            logger.warning(f"Finalization tasks failed for job {job.job_id}: {e}")

    async def get_job_status(self, job_id: str) -> Optional[ImportJob]:
        """Get job status and progress"""
        return self.active_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running import job"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        if job.status == ImportStatus.RUNNING:
            job.status = ImportStatus.CANCELLED
            logger.info(f"Import job {job_id} cancelled")
            return True

        return False

    async def pause_job(self, job_id: str) -> bool:
        """Pause a running import job"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        if job.status == ImportStatus.RUNNING:
            job.status = ImportStatus.PAUSED
            logger.info(f"Import job {job_id} paused")
            return True

        return False

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused import job"""
        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        if job.status == ImportStatus.PAUSED:
            job.status = ImportStatus.RUNNING
            # Continue processing
            await self._process_import_job(job)
            return True

        return False

    def add_progress_callback(self, callback: Callable[[ImportProgress], None]) -> None:
        """Add progress callback function"""
        self.progress_callbacks.append(callback)

    async def get_import_statistics(self) -> Dict[str, Any]:
        """Get import statistics"""
        active_jobs = len([j for j in self.active_jobs.values() if j.status == ImportStatus.RUNNING])
        completed_jobs = len([j for j in self.active_jobs.values() if j.status == ImportStatus.COMPLETED])
        failed_jobs = len([j for j in self.active_jobs.values() if j.status == ImportStatus.FAILED])

        total_processed = sum(j.progress.processed_records for j in self.active_jobs.values())
        total_successful = sum(j.progress.successful_records for j in self.active_jobs.values())

        return {
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "total_processed_records": total_processed,
            "total_successful_records": total_successful,
            "overall_success_rate": (total_successful / max(total_processed, 1)) * 100
        }

    async def close(self) -> None:
        """Close the bulk importer and cleanup resources"""
        try:
            # Cancel all running jobs
            for job_id, job in self.active_jobs.items():
                if job.status == ImportStatus.RUNNING:
                    await self.cancel_job(job_id)

            # Shutdown executors
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)

            if self.process_executor:
                self.process_executor.shutdown(wait=True)

            self.is_initialized = False
            logger.info("Bulk data importer closed")

        except Exception as e:
            logger.error(f"Error closing bulk importer: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global bulk importer instance
bulk_importer = None

async def get_bulk_importer(
    storage_service: HybridStorageService,
    cache_manager: RedisCacheManager
) -> BulkDataImporter:
    """Get or create global bulk importer instance"""
    global bulk_importer
    if bulk_importer is None:
        bulk_importer = BulkDataImporter(storage_service, cache_manager)
        await bulk_importer.initialize()
    return bulk_importer