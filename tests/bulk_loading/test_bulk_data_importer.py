"""
Tests for Bulk Data Importer
Tests bulk import functionality, parallel processing, and data validation
"""

import asyncio
import pytest
import tempfile
import json
import csv
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
import pandas as pd
from pathlib import Path

from src.bulk_loading.bulk_data_importer import (
    BulkDataImporter,
    ImportConfig,
    ImportFormat,
    ImportStatus,
    ImportJob,
    ImportProgress,
    ValidationLevel,
    DataValidator,
    DataTransformer,
    FileProcessor
)
from src.data_sources.base import PriceData


class TestImportConfig:
    """Test import configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ImportConfig()

        assert config.batch_size == 10000
        assert config.parallel_processing is True
        assert config.validation_level == ValidationLevel.STRICT
        assert config.skip_duplicates is True
        assert config.enable_progress_tracking is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = ImportConfig(
            batch_size=5000,
            validation_level=ValidationLevel.BASIC,
            parallel_processing=False,
            max_error_rate=0.1
        )

        assert config.batch_size == 5000
        assert config.validation_level == ValidationLevel.BASIC
        assert config.parallel_processing is False
        assert config.max_error_rate == 0.1


class TestImportProgress:
    """Test import progress tracking"""

    def test_progress_initialization(self):
        """Test progress initialization"""
        progress = ImportProgress("test_job_1")

        assert progress.job_id == "test_job_1"
        assert progress.total_records == 0
        assert progress.processed_records == 0
        assert progress.progress_percentage == 0.0
        assert progress.success_rate == 100.0

    def test_progress_calculations(self):
        """Test progress calculation properties"""
        progress = ImportProgress("test_job_1")
        progress.total_records = 1000
        progress.processed_records = 500
        progress.successful_records = 450
        progress.failed_records = 50

        assert progress.progress_percentage == 50.0
        assert progress.success_rate == 90.0

    def test_completion_estimation(self):
        """Test completion time estimation"""
        progress = ImportProgress("test_job_1")
        progress.total_records = 1000
        progress.processed_records = 250
        progress.start_time = datetime.now() - timedelta(minutes=5)

        progress.estimate_completion()

        assert progress.estimated_completion is not None
        assert progress.estimated_completion > datetime.now()


class TestImportJob:
    """Test import job functionality"""

    def test_job_initialization(self):
        """Test job initialization"""
        config = ImportConfig(batch_size=5000)
        job = ImportJob(
            job_id="test_job_1",
            name="Test Import",
            files=["test1.csv", "test2.csv"],
            format=ImportFormat.CSV,
            config=config
        )

        assert job.job_id == "test_job_1"
        assert job.name == "Test Import"
        assert job.status == ImportStatus.PENDING
        assert len(job.files) == 2
        assert job.format == ImportFormat.CSV
        assert job.config.batch_size == 5000

    def test_job_progress_initialization(self):
        """Test job progress is properly initialized"""
        job = ImportJob(
            job_id="test_job_1",
            name="Test Import",
            files=["test.csv"],
            format=ImportFormat.CSV,
            config=ImportConfig()
        )

        assert job.progress.job_id == "test_job_1"


class TestDataValidator:
    """Test data validation functionality"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return DataValidator(ValidationLevel.STRICT)

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataValidator(ValidationLevel.BASIC)
        assert validator.validation_level == ValidationLevel.BASIC

    def test_valid_record_validation(self, validator):
        """Test validation of valid record"""
        record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        is_valid, errors = validator.validate_record(record)
        assert is_valid
        assert len(errors) == 0

    def test_missing_fields_validation(self, validator):
        """Test validation with missing required fields"""
        record = {
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 155.0
            # Missing timestamp, low, close, volume
        }

        is_valid, errors = validator.validate_record(record)
        assert not is_valid
        assert len(errors) > 0
        assert any('timestamp' in error for error in errors)

    def test_invalid_prices_validation(self, validator):
        """Test validation with invalid price relationships"""
        record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 149.0,  # High less than open
            'low': 151.0,   # Low greater than open
            'close': 154.0,
            'volume': 1000000
        }

        is_valid, errors = validator.validate_record(record)
        assert not is_valid
        assert len(errors) > 0

    def test_negative_prices_validation(self, validator):
        """Test validation with negative prices"""
        record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': -150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        is_valid, errors = validator.validate_record(record)
        assert not is_valid
        assert any('positive' in error.lower() for error in errors)

    def test_invalid_symbol_validation(self, validator):
        """Test validation with invalid symbol"""
        record = {
            'symbol': 'invalid_symbol_123',  # Too long and contains numbers
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        is_valid, errors = validator.validate_record(record)
        assert not is_valid
        assert len(errors) > 0

    def test_custom_validator(self):
        """Test custom validator functionality"""
        validator = DataValidator(ValidationLevel.CUSTOM)

        def custom_rule(record):
            return record.get('volume', 0) > 1000

        validator.add_custom_validator(custom_rule)

        # Test record that passes custom validation
        valid_record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 2000
        }

        is_valid, errors = validator.validate_record(valid_record)
        assert is_valid

        # Test record that fails custom validation
        invalid_record = valid_record.copy()
        invalid_record['volume'] = 500

        is_valid, errors = validator.validate_record(invalid_record)
        assert not is_valid


class TestDataTransformer:
    """Test data transformation functionality"""

    @pytest.fixture
    def transformer(self):
        """Create transformer instance"""
        config = ImportConfig()
        return DataTransformer(config)

    def test_basic_transformation(self, transformer):
        """Test basic record transformation"""
        record = {
            'symbol': 'aapl',  # lowercase
            'timestamp': '2024-01-01T10:00:00',
            'open': '150.00',  # string
            'high': '155.00',
            'low': '149.00',
            'close': '154.00',
            'volume': '1000000'
        }

        price_data = transformer.transform_record(record)

        assert isinstance(price_data, PriceData)
        assert price_data.symbol == 'AAPL'  # Should be uppercase
        assert price_data.open_price == Decimal('150.00')
        assert price_data.volume == 1000000
        assert isinstance(price_data.timestamp, datetime)

    def test_timestamp_formats(self, transformer):
        """Test different timestamp format handling"""
        # ISO format
        record1 = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00Z',
            'open': 150.0, 'high': 155.0, 'low': 149.0, 'close': 154.0, 'volume': 1000000
        }

        price_data1 = transformer.transform_record(record1)
        assert isinstance(price_data1.timestamp, datetime)

        # Unix timestamp
        record2 = record1.copy()
        record2['timestamp'] = 1704105600  # 2024-01-01 10:00:00 UTC

        price_data2 = transformer.transform_record(record2)
        assert isinstance(price_data2.timestamp, datetime)

    def test_transformation_error_handling(self, transformer):
        """Test transformation error handling"""
        invalid_record = {
            'symbol': 'AAPL',
            'timestamp': 'invalid_timestamp',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        with pytest.raises(ValueError):
            transformer.transform_record(invalid_record)


class TestFileProcessor:
    """Test file processing functionality"""

    def test_csv_file_creation_and_reading(self):
        """Test CSV file processing"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for i in range(100):
                writer.writerow([
                    'AAPL',
                    f'2024-01-{i+1:02d}T10:00:00',
                    150.0 + i,
                    155.0 + i,
                    149.0 + i,
                    154.0 + i,
                    1000000 + i * 1000
                ])

            csv_file = f.name

        # Test reading
        async def test_reading():
            chunks = []
            async for chunk in FileProcessor.read_csv_file(csv_file, chunk_size=20):
                chunks.append(chunk)

            assert len(chunks) == 5  # 100 records / 20 chunk_size
            assert len(chunks[0]) == 20
            assert 'symbol' in chunks[0].columns

        asyncio.run(test_reading())

        # Cleanup
        Path(csv_file).unlink()

    def test_json_file_creation_and_reading(self):
        """Test JSON file processing"""
        # Create temporary JSON file
        test_data = []
        for i in range(50):
            test_data.append({
                'symbol': 'AAPL',
                'timestamp': f'2024-01-{i+1:02d}T10:00:00',
                'open': 150.0 + i,
                'high': 155.0 + i,
                'low': 149.0 + i,
                'close': 154.0 + i,
                'volume': 1000000 + i * 1000
            })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            json_file = f.name

        # Test reading
        async def test_reading():
            chunks = []
            async for chunk in FileProcessor.read_json_file(json_file, chunk_size=10):
                chunks.append(chunk)

            assert len(chunks) == 5  # 50 records / 10 chunk_size
            assert len(chunks[0]) == 10
            assert chunks[0][0]['symbol'] == 'AAPL'

        asyncio.run(test_reading())

        # Cleanup
        Path(json_file).unlink()


class TestBulkDataImporter:
    """Test bulk data importer functionality"""

    @pytest.fixture
    def mock_storage_service(self):
        """Mock storage service"""
        service = AsyncMock()
        service.bulk_insert_price_data.return_value = None
        return service

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager"""
        return AsyncMock()

    @pytest.fixture
    async def bulk_importer(self, mock_storage_service, mock_cache_manager):
        """Create bulk importer instance"""
        importer = BulkDataImporter(mock_storage_service, mock_cache_manager)
        await importer.initialize()
        return importer

    @pytest.mark.asyncio
    async def test_importer_initialization(self, mock_storage_service, mock_cache_manager):
        """Test importer initialization"""
        importer = BulkDataImporter(mock_storage_service, mock_cache_manager)
        assert not importer.is_initialized

        await importer.initialize()
        assert importer.is_initialized

        await importer.close()

    @pytest.mark.asyncio
    async def test_create_import_job(self, bulk_importer):
        """Test import job creation"""
        job_id = await bulk_importer.create_import_job(
            name="Test Import",
            files=["test1.csv", "test2.csv"],
            format=ImportFormat.CSV
        )

        assert job_id is not None
        assert job_id in bulk_importer.active_jobs

        job = bulk_importer.active_jobs[job_id]
        assert job.name == "Test Import"
        assert job.status == ImportStatus.PENDING
        assert len(job.files) == 2

    @pytest.mark.asyncio
    async def test_job_status_retrieval(self, bulk_importer):
        """Test job status retrieval"""
        job_id = await bulk_importer.create_import_job(
            name="Test Import",
            files=["test.csv"],
            format=ImportFormat.CSV
        )

        job = await bulk_importer.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.status == ImportStatus.PENDING

        # Test non-existent job
        missing_job = await bulk_importer.get_job_status("nonexistent")
        assert missing_job is None

    @pytest.mark.asyncio
    async def test_job_control_operations(self, bulk_importer):
        """Test job control operations (pause, cancel, resume)"""
        job_id = await bulk_importer.create_import_job(
            name="Test Import",
            files=["test.csv"],
            format=ImportFormat.CSV
        )

        # Test pause (job not running, should fail)
        result = await bulk_importer.pause_job(job_id)
        assert not result

        # Test cancel
        result = await bulk_importer.cancel_job(job_id)
        assert not result  # Not running, can't cancel

        # Test with non-existent job
        result = await bulk_importer.pause_job("nonexistent")
        assert not result

    @pytest.mark.asyncio
    async def test_import_statistics(self, bulk_importer):
        """Test import statistics collection"""
        # Create some jobs
        await bulk_importer.create_import_job("Job 1", ["test1.csv"], ImportFormat.CSV)
        await bulk_importer.create_import_job("Job 2", ["test2.csv"], ImportFormat.CSV)

        stats = await bulk_importer.get_import_statistics()

        assert 'active_jobs' in stats
        assert 'completed_jobs' in stats
        assert 'failed_jobs' in stats
        assert 'total_processed_records' in stats
        assert 'overall_success_rate' in stats

    @pytest.mark.asyncio
    async def test_record_processing_validation(self, bulk_importer):
        """Test record processing with validation"""
        # Create test data with valid and invalid records
        valid_record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        invalid_record = {
            'symbol': 'AAPL',
            'timestamp': '2024-01-01T10:00:00',
            'open': -150.0,  # Invalid negative price
            'high': 155.0,
            'low': 149.0,
            'close': 154.0,
            'volume': 1000000
        }

        # Test with individual records
        from src.bulk_loading.bulk_data_importer import DataValidator, DataTransformer, ImportConfig

        validator = DataValidator(ValidationLevel.STRICT)
        transformer = DataTransformer(ImportConfig())

        # Valid record should pass
        is_valid, errors = validator.validate_record(valid_record)
        assert is_valid

        if is_valid:
            price_data = transformer.transform_record(valid_record)
            assert isinstance(price_data, PriceData)

        # Invalid record should fail validation
        is_valid, errors = validator.validate_record(invalid_record)
        assert not is_valid
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_bulk_insert_integration(self, bulk_importer):
        """Test bulk insert integration with storage service"""
        # Create test price data
        price_data_list = []
        for i in range(10):
            price_data = PriceData(
                symbol='AAPL',
                timestamp=datetime.now() + timedelta(minutes=i),
                open_price=Decimal('150.00'),
                high_price=Decimal('155.00'),
                low_price=Decimal('149.00'),
                close_price=Decimal('154.00'),
                volume=1000000,
                source='test'
            )
            price_data_list.append(price_data)

        # Test bulk insert
        await bulk_importer._bulk_insert_records(price_data_list)

        # Verify storage service was called
        bulk_importer.storage_service.bulk_insert_price_data.assert_called_once_with(price_data_list)

    @pytest.mark.asyncio
    async def test_error_handling(self, bulk_importer):
        """Test error handling scenarios"""
        # Test with non-existent file
        job_id = await bulk_importer.create_import_job(
            name="Error Test",
            files=["nonexistent_file.csv"],
            format=ImportFormat.CSV
        )

        # Starting job with non-existent file should handle error gracefully
        result = await bulk_importer.start_import_job(job_id)
        assert not result

        job = await bulk_importer.get_job_status(job_id)
        assert job.status == ImportStatus.FAILED

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_storage_service, mock_cache_manager):
        """Test bulk importer as context manager"""
        async with BulkDataImporter(mock_storage_service, mock_cache_manager) as importer:
            assert importer.is_initialized

        # Should be closed after context exit
        assert not importer.is_initialized

    @pytest.mark.asyncio
    async def test_progress_callbacks(self, bulk_importer):
        """Test progress callback functionality"""
        callback_called = False
        callback_progress = None

        async def progress_callback(progress):
            nonlocal callback_called, callback_progress
            callback_called = True
            callback_progress = progress

        bulk_importer.add_progress_callback(progress_callback)

        # Create and start a job
        job_id = await bulk_importer.create_import_job(
            name="Callback Test",
            files=["test.csv"],
            format=ImportFormat.CSV
        )

        # Verify callback was added
        assert len(bulk_importer.progress_callbacks) == 1


class TestImportFormats:
    """Test different import formats"""

    def test_import_format_enum(self):
        """Test import format enumeration"""
        assert ImportFormat.CSV.value == "csv"
        assert ImportFormat.JSON.value == "json"
        assert ImportFormat.PARQUET.value == "parquet"
        assert ImportFormat.XLSX.value == "xlsx"

    def test_validation_level_enum(self):
        """Test validation level enumeration"""
        assert ValidationLevel.NONE.value == "none"
        assert ValidationLevel.BASIC.value == "basic"
        assert ValidationLevel.STRICT.value == "strict"
        assert ValidationLevel.CUSTOM.value == "custom"

    def test_import_status_enum(self):
        """Test import status enumeration"""
        assert ImportStatus.PENDING.value == "pending"
        assert ImportStatus.RUNNING.value == "running"
        assert ImportStatus.COMPLETED.value == "completed"
        assert ImportStatus.FAILED.value == "failed"


class TestBulkImportIntegration:
    """Integration tests for bulk import system"""

    @pytest.mark.integration
    async def test_full_import_workflow(self):
        """Integration test for complete import workflow"""
        # This test would require real file creation and processing
        pytest.skip("Integration test requires real file system and database")

    @pytest.mark.integration
    async def test_large_file_processing(self):
        """Test processing of large files"""
        # This test would create and process large test files
        pytest.skip("Integration test requires large file processing setup")

    @pytest.mark.integration
    async def test_parallel_processing_performance(self):
        """Test parallel processing performance"""
        # This test would measure actual performance with parallel workers
        pytest.skip("Integration test requires performance measurement setup")

    @pytest.mark.integration
    async def test_error_recovery_and_resumption(self):
        """Test error recovery and job resumption"""
        # This test would simulate errors and test recovery mechanisms
        pytest.skip("Integration test requires error simulation setup")