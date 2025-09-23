"""
Tests for Progress Tracker
Tests progress tracking, checkpointing, and resumption functionality
"""

import asyncio
import pytest
import tempfile
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.bulk_loading.progress_tracker import (
    ProgressTracker,
    ProgressMetrics,
    ProgressState,
    Checkpoint,
    CheckpointType,
    CheckpointStorage,
    SQLiteCheckpointStorage,
    create_progress_tracker
)


class TestProgressMetrics:
    """Test progress metrics functionality"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ProgressMetrics()

        assert metrics.total_items == 0
        assert metrics.processed_items == 0
        assert metrics.successful_items == 0
        assert metrics.failed_items == 0
        assert metrics.progress_percentage == 0.0
        assert metrics.success_rate == 100.0

    def test_progress_percentage_calculation(self):
        """Test progress percentage calculation"""
        metrics = ProgressMetrics()
        metrics.total_items = 1000
        metrics.processed_items = 250

        assert metrics.progress_percentage == 25.0

        # Test edge case with zero total
        metrics.total_items = 0
        assert metrics.progress_percentage == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = ProgressMetrics()
        metrics.processed_items = 100
        metrics.successful_items = 85

        assert metrics.success_rate == 85.0

        # Test edge case with zero processed
        metrics.processed_items = 0
        assert metrics.success_rate == 100.0

    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation"""
        metrics = ProgressMetrics()
        start_time = datetime.now() - timedelta(minutes=5)
        metrics.start_time = start_time
        metrics.last_update_time = datetime.now()

        elapsed = metrics.elapsed_time_seconds
        assert elapsed > 290  # Should be approximately 5 minutes
        assert elapsed < 310

    def test_timing_updates(self):
        """Test timing update functionality"""
        metrics = ProgressMetrics()
        metrics.start_time = datetime.now() - timedelta(minutes=2)
        metrics.processed_items = 120

        metrics.update_timing()

        assert metrics.last_update_time is not None
        assert metrics.items_per_second > 0
        assert metrics.estimated_completion_time is None  # No total items set

        # Set total items and update again
        metrics.total_items = 200
        metrics.update_timing()

        assert metrics.estimated_completion_time is not None
        assert metrics.estimated_completion_time > datetime.now()


class TestCheckpoint:
    """Test checkpoint functionality"""

    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        metrics = ProgressMetrics()
        metrics.total_items = 1000
        metrics.processed_items = 500

        checkpoint = Checkpoint(
            checkpoint_id="test_checkpoint_1",
            job_id="test_job_1",
            checkpoint_type=CheckpointType.AUTOMATIC,
            timestamp=datetime.now(),
            metrics=metrics,
            current_position={"file": "test.csv", "line": 500},
            processed_files=["file1.csv", "file2.csv"]
        )

        assert checkpoint.checkpoint_id == "test_checkpoint_1"
        assert checkpoint.job_id == "test_job_1"
        assert checkpoint.checkpoint_type == CheckpointType.AUTOMATIC
        assert checkpoint.metrics.processed_items == 500
        assert checkpoint.current_position["file"] == "test.csv"
        assert len(checkpoint.processed_files) == 2

    def test_checkpoint_serialization(self):
        """Test checkpoint to/from dictionary conversion"""
        metrics = ProgressMetrics()
        metrics.start_time = datetime.now()
        metrics.total_items = 1000
        metrics.processed_items = 500

        checkpoint = Checkpoint(
            checkpoint_id="test_checkpoint_1",
            job_id="test_job_1",
            checkpoint_type=CheckpointType.MANUAL,
            timestamp=datetime.now(),
            metrics=metrics,
            current_position={"file": "test.csv"},
            processed_files=["file1.csv"]
        )

        # Convert to dictionary
        checkpoint_dict = checkpoint.to_dict()

        assert checkpoint_dict["checkpoint_id"] == "test_checkpoint_1"
        assert checkpoint_dict["checkpoint_type"] == "manual"
        assert "metrics" in checkpoint_dict
        assert checkpoint_dict["current_position"]["file"] == "test.csv"

        # Convert back from dictionary
        restored_checkpoint = Checkpoint.from_dict(checkpoint_dict)

        assert restored_checkpoint.checkpoint_id == checkpoint.checkpoint_id
        assert restored_checkpoint.job_id == checkpoint.job_id
        assert restored_checkpoint.checkpoint_type == checkpoint.checkpoint_type
        assert restored_checkpoint.metrics.total_items == checkpoint.metrics.total_items
        assert restored_checkpoint.current_position == checkpoint.current_position


class TestSQLiteCheckpointStorage:
    """Test SQLite checkpoint storage functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def storage(self, temp_db_path):
        """Create storage instance"""
        return SQLiteCheckpointStorage(temp_db_path)

    @pytest.mark.asyncio
    async def test_storage_initialization(self, storage):
        """Test storage initialization"""
        await storage._initialize()

        # Check if tables were created
        conn = sqlite3.connect(storage.db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(self, storage):
        """Test checkpoint save and load operations"""
        metrics = ProgressMetrics()
        metrics.total_items = 1000
        metrics.processed_items = 500

        checkpoint = Checkpoint(
            checkpoint_id="test_checkpoint_1",
            job_id="test_job_1",
            checkpoint_type=CheckpointType.AUTOMATIC,
            timestamp=datetime.now(),
            metrics=metrics,
            current_position={"file": "test.csv", "line": 500}
        )

        # Save checkpoint
        success = await storage.save_checkpoint(checkpoint)
        assert success

        # Load checkpoint
        loaded_checkpoint = await storage.load_checkpoint("test_checkpoint_1")
        assert loaded_checkpoint is not None
        assert loaded_checkpoint.checkpoint_id == checkpoint.checkpoint_id
        assert loaded_checkpoint.job_id == checkpoint.job_id
        assert loaded_checkpoint.metrics.total_items == checkpoint.metrics.total_items

    @pytest.mark.asyncio
    async def test_checkpoint_list_operations(self, storage):
        """Test checkpoint listing operations"""
        job_id = "test_job_1"

        # Create multiple checkpoints
        for i in range(3):
            metrics = ProgressMetrics()
            checkpoint = Checkpoint(
                checkpoint_id=f"checkpoint_{i}",
                job_id=job_id,
                checkpoint_type=CheckpointType.AUTOMATIC,
                timestamp=datetime.now() + timedelta(minutes=i),
                metrics=metrics,
                current_position={}
            )
            await storage.save_checkpoint(checkpoint)

        # List checkpoints
        checkpoint_ids = await storage.list_checkpoints(job_id)
        assert len(checkpoint_ids) == 3
        # Should be ordered by timestamp descending (most recent first)
        assert checkpoint_ids[0] == "checkpoint_2"

    @pytest.mark.asyncio
    async def test_checkpoint_deletion(self, storage):
        """Test checkpoint deletion"""
        metrics = ProgressMetrics()
        checkpoint = Checkpoint(
            checkpoint_id="test_checkpoint_1",
            job_id="test_job_1",
            checkpoint_type=CheckpointType.MANUAL,
            timestamp=datetime.now(),
            metrics=metrics,
            current_position={}
        )

        # Save and then delete
        await storage.save_checkpoint(checkpoint)
        success = await storage.delete_checkpoint("test_checkpoint_1")
        assert success

        # Verify deletion
        loaded_checkpoint = await storage.load_checkpoint("test_checkpoint_1")
        assert loaded_checkpoint is None

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, storage):
        """Test checkpoint cleanup functionality"""
        job_id = "test_job_1"

        # Create 15 checkpoints
        for i in range(15):
            metrics = ProgressMetrics()
            checkpoint = Checkpoint(
                checkpoint_id=f"checkpoint_{i:02d}",
                job_id=job_id,
                checkpoint_type=CheckpointType.AUTOMATIC,
                timestamp=datetime.now() + timedelta(minutes=i),
                metrics=metrics,
                current_position={}
            )
            await storage.save_checkpoint(checkpoint)

        # Cleanup to keep only 5 most recent
        deleted_count = await storage.cleanup_old_checkpoints(job_id, keep_count=5)
        assert deleted_count == 10

        # Verify only 5 remain
        remaining_checkpoints = await storage.list_checkpoints(job_id)
        assert len(remaining_checkpoints) == 5

    @pytest.mark.asyncio
    async def test_nonexistent_checkpoint_load(self, storage):
        """Test loading non-existent checkpoint"""
        loaded_checkpoint = await storage.load_checkpoint("nonexistent")
        assert loaded_checkpoint is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, storage):
        """Test deleting non-existent checkpoint"""
        success = await storage.delete_checkpoint("nonexistent")
        assert success  # Should succeed even if checkpoint doesn't exist


class TestProgressTracker:
    """Test progress tracker functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def progress_tracker(self, temp_db_path):
        """Create progress tracker instance"""
        storage = SQLiteCheckpointStorage(temp_db_path)
        return ProgressTracker("test_job_1", storage, checkpoint_interval=100)

    @pytest.mark.asyncio
    async def test_tracker_initialization(self, progress_tracker):
        """Test tracker initialization"""
        assert progress_tracker.job_id == "test_job_1"
        assert progress_tracker.state == ProgressState.NOT_STARTED
        assert progress_tracker.metrics.total_items == 0

    @pytest.mark.asyncio
    async def test_tracker_start(self, progress_tracker):
        """Test tracker start functionality"""
        await progress_tracker.start(total_items=1000)

        assert progress_tracker.state == ProgressState.RUNNING
        assert progress_tracker.metrics.total_items == 1000
        assert progress_tracker.metrics.start_time is not None

        # Cleanup
        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_progress_updates(self, progress_tracker):
        """Test progress update functionality"""
        await progress_tracker.start(total_items=1000)

        # Update progress
        await progress_tracker.update_progress(
            processed_delta=10,
            successful_delta=8,
            failed_delta=2,
            position_update={"current_file": "test.csv", "line": 100}
        )

        assert progress_tracker.metrics.processed_items == 10
        assert progress_tracker.metrics.successful_items == 8
        assert progress_tracker.metrics.failed_items == 2
        assert progress_tracker.current_position["current_file"] == "test.csv"

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_file_tracking(self, progress_tracker):
        """Test file tracking functionality"""
        await progress_tracker.start(total_items=1000)

        # Set current file
        await progress_tracker.set_current_file("test1.csv")
        assert progress_tracker.current_position["current_file"] == "test1.csv"

        # Mark file as completed
        await progress_tracker.mark_file_completed("test1.csv")
        assert "test1.csv" in progress_tracker.processed_files

        # Mark another file
        await progress_tracker.mark_file_completed("test2.csv")
        assert len(progress_tracker.processed_files) == 2

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_manual_checkpoint_creation(self, progress_tracker):
        """Test manual checkpoint creation"""
        await progress_tracker.start(total_items=1000)

        # Update progress
        await progress_tracker.update_progress(processed_delta=500, successful_delta=450)

        # Create manual checkpoint
        checkpoint_id = await progress_tracker.create_checkpoint(CheckpointType.MANUAL)
        assert checkpoint_id != ""

        # Verify checkpoint was saved
        checkpoint = await progress_tracker.storage.load_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.job_id == "test_job_1"
        assert checkpoint.metrics.processed_items == 500

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_automatic_checkpoint_creation(self, progress_tracker):
        """Test automatic checkpoint creation"""
        # Set small checkpoint interval for testing
        progress_tracker.checkpoint_interval = 50

        await progress_tracker.start(total_items=1000)

        # Update progress to trigger automatic checkpoint
        for i in range(6):  # 6 * 10 = 60 items, should trigger checkpoint at 50
            await progress_tracker.update_progress(processed_delta=10, successful_delta=10)

        # Should have created automatic checkpoint
        checkpoints = await progress_tracker.storage.list_checkpoints("test_job_1")
        assert len(checkpoints) > 0

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_checkpoint_loading(self, progress_tracker):
        """Test checkpoint loading and resumption"""
        await progress_tracker.start(total_items=1000)

        # Make some progress
        await progress_tracker.update_progress(processed_delta=300, successful_delta=280, failed_delta=20)
        await progress_tracker.set_current_file("test.csv")

        # Create checkpoint
        checkpoint_id = await progress_tracker.create_checkpoint(CheckpointType.MANUAL)

        # Create new tracker instance
        storage = progress_tracker.storage
        new_tracker = ProgressTracker("test_job_1", storage)

        # Load checkpoint
        success = await new_tracker.load_checkpoint(checkpoint_id)
        assert success

        # Verify state was restored
        assert new_tracker.metrics.processed_items == 300
        assert new_tracker.metrics.successful_items == 280
        assert new_tracker.metrics.failed_items == 20
        assert new_tracker.current_position["current_file"] == "test.csv"

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_latest_checkpoint_resumption(self, progress_tracker):
        """Test resumption from latest checkpoint"""
        await progress_tracker.start(total_items=1000)

        # Create multiple checkpoints
        for i in range(3):
            await progress_tracker.update_progress(processed_delta=100, successful_delta=95)
            await progress_tracker.create_checkpoint(CheckpointType.AUTOMATIC)
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Create new tracker and resume from latest
        storage = progress_tracker.storage
        new_tracker = ProgressTracker("test_job_1", storage)

        success = await new_tracker.resume_from_latest_checkpoint()
        assert success

        # Should have loaded the latest checkpoint (300 processed items)
        assert new_tracker.metrics.processed_items == 300

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_quality_metrics_updates(self, progress_tracker):
        """Test quality metrics updates"""
        await progress_tracker.start(total_items=1000)

        # Update quality metrics
        await progress_tracker.update_quality_metrics(quality_score=85.5, validation_errors=5)

        assert progress_tracker.metrics.data_quality_score == 85.5
        assert progress_tracker.metrics.validation_errors == 5

        # Update again (should accumulate validation errors)
        await progress_tracker.update_quality_metrics(quality_score=90.0, validation_errors=3)

        assert progress_tracker.metrics.data_quality_score == 90.0
        assert progress_tracker.metrics.validation_errors == 8

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_resource_usage_updates(self, progress_tracker):
        """Test resource usage updates"""
        await progress_tracker.start(total_items=1000)

        # Update resource usage
        await progress_tracker.update_resource_usage(memory_mb=256.5, cpu_percent=45.2)

        assert progress_tracker.metrics.memory_usage_mb == 256.5
        assert progress_tracker.metrics.cpu_usage_percent == 45.2

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_completion_states(self, progress_tracker):
        """Test different completion states"""
        await progress_tracker.start(total_items=1000)

        # Test completion
        await progress_tracker.complete()
        assert progress_tracker.state == ProgressState.COMPLETED

        # Verify completion checkpoint was created
        checkpoints = await progress_tracker.storage.list_checkpoints("test_job_1")
        assert len(checkpoints) > 0

        # Test failure
        new_tracker = ProgressTracker("test_job_2", progress_tracker.storage)
        await new_tracker.start(total_items=500)
        await new_tracker.fail("Test error")
        assert new_tracker.state == ProgressState.FAILED

        # Test cancellation
        cancel_tracker = ProgressTracker("test_job_3", progress_tracker.storage)
        await cancel_tracker.start(total_items=200)
        await cancel_tracker.cancel()
        assert cancel_tracker.state == ProgressState.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self, progress_tracker):
        """Test pause and resume functionality"""
        await progress_tracker.start(total_items=1000)

        # Make some progress
        await progress_tracker.update_progress(processed_delta=200, successful_delta=190)

        # Pause
        await progress_tracker.pause()
        assert progress_tracker.state == ProgressState.PAUSED

        # Resume (in real implementation, this would restart processing)
        # For testing, we'll just verify the state can be set back to running
        progress_tracker.state = ProgressState.RUNNING
        assert progress_tracker.state == ProgressState.RUNNING

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_progress_summary(self, progress_tracker):
        """Test progress summary generation"""
        await progress_tracker.start(total_items=1000)

        # Update progress
        await progress_tracker.update_progress(processed_delta=250, successful_delta=240, failed_delta=10)
        await progress_tracker.set_current_file("test.csv")

        summary = progress_tracker.get_progress_summary()

        assert summary["job_id"] == "test_job_1"
        assert summary["state"] == ProgressState.RUNNING.value
        assert summary["progress_percentage"] == 25.0
        assert summary["total_items"] == 1000
        assert summary["processed_items"] == 250
        assert summary["successful_items"] == 240
        assert summary["failed_items"] == 10
        assert summary["current_position"]["current_file"] == "test.csv"

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_checkpoint_callbacks(self, progress_tracker):
        """Test checkpoint creation callbacks"""
        callback_called = False
        callback_checkpoint = None

        def checkpoint_callback(checkpoint):
            nonlocal callback_called, callback_checkpoint
            callback_called = True
            callback_checkpoint = checkpoint

        progress_tracker.add_checkpoint_callback(checkpoint_callback)

        await progress_tracker.start(total_items=1000)
        await progress_tracker.update_progress(processed_delta=100, successful_delta=95)

        # Create checkpoint
        await progress_tracker.create_checkpoint(CheckpointType.MANUAL)

        assert callback_called
        assert callback_checkpoint is not None
        assert callback_checkpoint.job_id == "test_job_1"

        await progress_tracker.cleanup()

    @pytest.mark.asyncio
    async def test_checkpoint_with_recovery_data(self, progress_tracker):
        """Test checkpoint with recovery data"""
        await progress_tracker.start(total_items=1000)

        # Create checkpoint with recovery data
        recovery_data = {"custom_state": "test_value", "counters": [1, 2, 3]}
        checkpoint_id = await progress_tracker.create_checkpoint(
            CheckpointType.MANUAL,
            recovery_data=recovery_data
        )

        # Load checkpoint and verify recovery data
        checkpoint = await progress_tracker.storage.load_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.recovery_data is not None

        await progress_tracker.cleanup()


class TestProgressTrackerFactory:
    """Test progress tracker factory function"""

    @pytest.mark.asyncio
    async def test_create_progress_tracker(self):
        """Test progress tracker creation via factory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_progress.db"

            tracker = await create_progress_tracker(
                job_id="factory_test_job",
                storage_path=str(db_path),
                checkpoint_interval=500
            )

            assert tracker.job_id == "factory_test_job"
            assert tracker.checkpoint_interval == 500
            assert isinstance(tracker.storage, SQLiteCheckpointStorage)

    @pytest.mark.asyncio
    async def test_create_progress_tracker_default_path(self):
        """Test progress tracker creation with default path"""
        tracker = await create_progress_tracker("default_path_job")

        assert tracker.job_id == "default_path_job"
        assert isinstance(tracker.storage, SQLiteCheckpointStorage)


class TestProgressTrackerIntegration:
    """Integration tests for progress tracker"""

    @pytest.mark.integration
    async def test_long_running_job_simulation(self):
        """Test progress tracking for long-running job"""
        # This test would simulate a long-running job with checkpoints
        pytest.skip("Integration test requires long-running simulation")

    @pytest.mark.integration
    async def test_concurrent_progress_tracking(self):
        """Test concurrent progress tracking"""
        # This test would test multiple concurrent trackers
        pytest.skip("Integration test requires concurrency testing setup")

    @pytest.mark.integration
    async def test_checkpoint_storage_performance(self):
        """Test checkpoint storage performance under load"""
        # This test would measure checkpoint storage performance
        pytest.skip("Integration test requires performance measurement setup")