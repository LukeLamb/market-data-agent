"""
Progress Tracking and Resumption System for Bulk Data Loading
Provides detailed progress monitoring, checkpointing, and resumption capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
import pickle
import sqlite3
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Types of checkpoints"""
    AUTOMATIC = "automatic"     # Periodic automatic checkpoints
    MANUAL = "manual"          # User-triggered checkpoints
    ERROR = "error"            # Error recovery checkpoints
    COMPLETION = "completion"   # Final completion checkpoint


class ProgressState(Enum):
    """Progress tracking states"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressMetrics:
    """Detailed progress metrics"""
    # Basic counters
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0

    # Timing information
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None

    # Performance metrics
    items_per_second: float = 0.0
    average_processing_time_ms: float = 0.0
    peak_processing_rate: float = 0.0

    # Memory and resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Quality metrics
    data_quality_score: float = 0.0
    validation_errors: int = 0

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.processed_items == 0:
            return 100.0
        return (self.successful_items / self.processed_items) * 100

    @property
    def elapsed_time_seconds(self) -> float:
        """Calculate elapsed time in seconds"""
        if not self.start_time:
            return 0.0
        end_time = self.last_update_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    def update_timing(self):
        """Update timing-related metrics"""
        self.last_update_time = datetime.now()

        if self.start_time:
            elapsed = self.elapsed_time_seconds
            if elapsed > 0 and self.processed_items > 0:
                self.items_per_second = self.processed_items / elapsed

                # Update peak rate
                if self.items_per_second > self.peak_processing_rate:
                    self.peak_processing_rate = self.items_per_second

                # Estimate completion time
                if self.total_items > self.processed_items:
                    remaining_items = self.total_items - self.processed_items
                    remaining_seconds = remaining_items / self.items_per_second
                    self.estimated_completion_time = datetime.now() + timedelta(seconds=remaining_seconds)


@dataclass
class Checkpoint:
    """Represents a progress checkpoint"""
    checkpoint_id: str
    job_id: str
    checkpoint_type: CheckpointType
    timestamp: datetime

    # Progress state
    metrics: ProgressMetrics
    current_position: Dict[str, Any]  # Current processing position
    processed_files: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Recovery information
    recovery_data: Optional[bytes] = None  # Serialized state for recovery

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'job_id': self.job_id,
            'checkpoint_type': self.checkpoint_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metrics': asdict(self.metrics),
            'current_position': self.current_position,
            'processed_files': self.processed_files,
            'metadata': self.metadata,
            'recovery_data': self.recovery_data.hex() if self.recovery_data else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary"""
        metrics_data = data.get('metrics', {})

        # Handle datetime fields in metrics
        if 'start_time' in metrics_data and metrics_data['start_time']:
            metrics_data['start_time'] = datetime.fromisoformat(metrics_data['start_time'])
        if 'last_update_time' in metrics_data and metrics_data['last_update_time']:
            metrics_data['last_update_time'] = datetime.fromisoformat(metrics_data['last_update_time'])
        if 'estimated_completion_time' in metrics_data and metrics_data['estimated_completion_time']:
            metrics_data['estimated_completion_time'] = datetime.fromisoformat(metrics_data['estimated_completion_time'])

        metrics = ProgressMetrics(**metrics_data)

        recovery_data = None
        if data.get('recovery_data'):
            recovery_data = bytes.fromhex(data['recovery_data'])

        return cls(
            checkpoint_id=data['checkpoint_id'],
            job_id=data['job_id'],
            checkpoint_type=CheckpointType(data['checkpoint_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metrics=metrics,
            current_position=data.get('current_position', {}),
            processed_files=data.get('processed_files', []),
            metadata=data.get('metadata', {}),
            recovery_data=recovery_data
        )


class CheckpointStorage:
    """Storage interface for checkpoints"""

    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint to storage"""
        raise NotImplementedError

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from storage"""
        raise NotImplementedError

    async def list_checkpoints(self, job_id: str) -> List[str]:
        """List checkpoint IDs for a job"""
        raise NotImplementedError

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from storage"""
        raise NotImplementedError

    async def cleanup_old_checkpoints(self, job_id: str, keep_count: int = 10) -> int:
        """Clean up old checkpoints, keeping only the most recent ones"""
        raise NotImplementedError


class SQLiteCheckpointStorage(CheckpointStorage):
    """SQLite-based checkpoint storage"""

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialized = False

    async def _initialize(self):
        """Initialize database schema"""
        if self._initialized:
            return

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        checkpoint_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        data TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_job_id ON checkpoints(job_id)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)
                """)

                conn.commit()
                self._initialized = True
            finally:
                conn.close()

    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint to SQLite database"""
        await self._initialize()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    checkpoint_data = json.dumps(checkpoint.to_dict())

                    conn.execute("""
                        INSERT OR REPLACE INTO checkpoints
                        (checkpoint_id, job_id, checkpoint_type, timestamp, data)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        checkpoint.checkpoint_id,
                        checkpoint.job_id,
                        checkpoint.checkpoint_type.value,
                        checkpoint.timestamp.isoformat(),
                        checkpoint_data
                    ))

                    conn.commit()
                    return True
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            return False

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from SQLite database"""
        await self._initialize()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.execute(
                        "SELECT data FROM checkpoints WHERE checkpoint_id = ?",
                        (checkpoint_id,)
                    )

                    row = cursor.fetchone()
                    if row:
                        checkpoint_data = json.loads(row[0])
                        return Checkpoint.from_dict(checkpoint_data)

                    return None
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def list_checkpoints(self, job_id: str) -> List[str]:
        """List checkpoint IDs for a job"""
        await self._initialize()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.execute("""
                        SELECT checkpoint_id FROM checkpoints
                        WHERE job_id = ?
                        ORDER BY timestamp DESC
                    """, (job_id,))

                    return [row[0] for row in cursor.fetchall()]
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Failed to list checkpoints for job {job_id}: {e}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from database"""
        await self._initialize()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    conn.execute(
                        "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                        (checkpoint_id,)
                    )

                    conn.commit()
                    return True
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def cleanup_old_checkpoints(self, job_id: str, keep_count: int = 10) -> int:
        """Clean up old checkpoints, keeping only the most recent ones"""
        await self._initialize()

        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    # Get checkpoints to delete (keep only the most recent ones)
                    cursor = conn.execute("""
                        SELECT checkpoint_id FROM checkpoints
                        WHERE job_id = ?
                        ORDER BY timestamp DESC
                        LIMIT -1 OFFSET ?
                    """, (job_id, keep_count))

                    checkpoints_to_delete = [row[0] for row in cursor.fetchall()]

                    # Delete old checkpoints
                    for checkpoint_id in checkpoints_to_delete:
                        conn.execute(
                            "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                            (checkpoint_id,)
                        )

                    conn.commit()
                    return len(checkpoints_to_delete)
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints for job {job_id}: {e}")
            return 0


class ProgressTracker:
    """Main progress tracking system"""

    def __init__(
        self,
        job_id: str,
        storage: CheckpointStorage = None,
        checkpoint_interval: int = 1000,  # Items between auto checkpoints
        max_checkpoints: int = 20
    ):
        self.job_id = job_id
        self.storage = storage or SQLiteCheckpointStorage()
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints

        # Progress state
        self.metrics = ProgressMetrics()
        self.state = ProgressState.NOT_STARTED
        self.current_position: Dict[str, Any] = {}
        self.processed_files: List[str] = []

        # Checkpointing
        self.last_checkpoint_count = 0
        self.checkpoint_callbacks: List[Callable[[Checkpoint], None]] = []

        # Recovery data
        self._recovery_data: Dict[str, Any] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Background checkpoint task
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self, total_items: int) -> None:
        """Start progress tracking"""
        with self._lock:
            self.metrics.total_items = total_items
            self.metrics.start_time = datetime.now()
            self.metrics.update_timing()
            self.state = ProgressState.RUNNING

        # Start background checkpoint task
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        logger.info(f"Progress tracking started for job {self.job_id} with {total_items} items")

    async def update_progress(
        self,
        processed_delta: int = 1,
        successful_delta: int = 0,
        failed_delta: int = 0,
        skipped_delta: int = 0,
        position_update: Dict[str, Any] = None
    ) -> None:
        """Update progress metrics"""
        with self._lock:
            self.metrics.processed_items += processed_delta
            self.metrics.successful_items += successful_delta
            self.metrics.failed_items += failed_delta
            self.metrics.skipped_items += skipped_delta

            if position_update:
                self.current_position.update(position_update)

            self.metrics.update_timing()

            # Check for automatic checkpoint
            items_since_checkpoint = self.metrics.processed_items - self.last_checkpoint_count
            if items_since_checkpoint >= self.checkpoint_interval:
                await self._create_automatic_checkpoint()

    async def set_current_file(self, file_path: str) -> None:
        """Set currently processing file"""
        with self._lock:
            self.current_position['current_file'] = file_path

    async def mark_file_completed(self, file_path: str) -> None:
        """Mark file as completed"""
        with self._lock:
            if file_path not in self.processed_files:
                self.processed_files.append(file_path)

    async def update_quality_metrics(self, quality_score: float, validation_errors: int = 0) -> None:
        """Update data quality metrics"""
        with self._lock:
            self.metrics.data_quality_score = quality_score
            self.metrics.validation_errors += validation_errors

    async def update_resource_usage(self, memory_mb: float, cpu_percent: float) -> None:
        """Update resource usage metrics"""
        with self._lock:
            self.metrics.memory_usage_mb = memory_mb
            self.metrics.cpu_usage_percent = cpu_percent

    async def create_checkpoint(self, checkpoint_type: CheckpointType = CheckpointType.MANUAL, recovery_data: Any = None) -> str:
        """Create a checkpoint"""
        checkpoint_id = self._generate_checkpoint_id()

        # Serialize recovery data if provided
        serialized_recovery_data = None
        if recovery_data is not None:
            try:
                serialized_recovery_data = pickle.dumps(recovery_data)
            except Exception as e:
                logger.warning(f"Failed to serialize recovery data: {e}")

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            job_id=self.job_id,
            checkpoint_type=checkpoint_type,
            timestamp=datetime.now(),
            metrics=ProgressMetrics(**asdict(self.metrics)),  # Create copy
            current_position=dict(self.current_position),  # Create copy
            processed_files=list(self.processed_files),  # Create copy
            recovery_data=serialized_recovery_data
        )

        # Save checkpoint
        success = await self.storage.save_checkpoint(checkpoint)

        if success:
            logger.info(f"Created checkpoint {checkpoint_id} for job {self.job_id}")

            # Update last checkpoint count for automatic checkpointing
            if checkpoint_type == CheckpointType.AUTOMATIC:
                self.last_checkpoint_count = self.metrics.processed_items

            # Notify callbacks
            for callback in self.checkpoint_callbacks:
                try:
                    callback(checkpoint)
                except Exception as e:
                    logger.warning(f"Checkpoint callback failed: {e}")

            # Cleanup old checkpoints
            await self.storage.cleanup_old_checkpoints(self.job_id, self.max_checkpoints)

            return checkpoint_id
        else:
            logger.error(f"Failed to save checkpoint for job {self.job_id}")
            return ""

    async def _create_automatic_checkpoint(self) -> None:
        """Create automatic checkpoint"""
        await self.create_checkpoint(CheckpointType.AUTOMATIC)

    async def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load progress from checkpoint"""
        checkpoint = await self.storage.load_checkpoint(checkpoint_id)

        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        if checkpoint.job_id != self.job_id:
            logger.error(f"Checkpoint {checkpoint_id} belongs to different job")
            return False

        with self._lock:
            # Restore metrics
            self.metrics = checkpoint.metrics
            self.current_position = checkpoint.current_position
            self.processed_files = checkpoint.processed_files

            # Update tracking state
            self.last_checkpoint_count = self.metrics.processed_items
            self.state = ProgressState.RUNNING

        logger.info(f"Loaded checkpoint {checkpoint_id} for job {self.job_id}")
        return True

    async def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the latest checkpoint for the job"""
        checkpoint_ids = await self.storage.list_checkpoints(self.job_id)

        if not checkpoint_ids:
            return None

        return await self.storage.load_checkpoint(checkpoint_ids[0])

    async def resume_from_latest_checkpoint(self) -> bool:
        """Resume from the latest checkpoint"""
        latest_checkpoint = await self.get_latest_checkpoint()

        if not latest_checkpoint:
            logger.info(f"No checkpoints found for job {self.job_id}")
            return False

        return await self.load_checkpoint(latest_checkpoint.checkpoint_id)

    def get_recovery_data(self, checkpoint_id: str) -> Any:
        """Get recovery data from checkpoint"""
        # This would be implemented to deserialize recovery data
        # For now, return None
        return None

    async def complete(self) -> None:
        """Mark progress as completed"""
        with self._lock:
            self.state = ProgressState.COMPLETED
            self.metrics.update_timing()

        # Create final checkpoint
        await self.create_checkpoint(CheckpointType.COMPLETION)

        # Stop background tasks
        self._stop_event.set()
        if self._checkpoint_task:
            await self._checkpoint_task

        logger.info(f"Progress tracking completed for job {self.job_id}")

    async def fail(self, error: str) -> None:
        """Mark progress as failed"""
        with self._lock:
            self.state = ProgressState.FAILED
            self.metrics.update_timing()

        # Create error checkpoint
        await self.create_checkpoint(CheckpointType.ERROR)

        logger.error(f"Progress tracking failed for job {self.job_id}: {error}")

    async def pause(self) -> None:
        """Pause progress tracking"""
        with self._lock:
            self.state = ProgressState.PAUSED

        # Create checkpoint
        await self.create_checkpoint(CheckpointType.MANUAL)

        logger.info(f"Progress tracking paused for job {self.job_id}")

    async def cancel(self) -> None:
        """Cancel progress tracking"""
        with self._lock:
            self.state = ProgressState.CANCELLED

        # Stop background tasks
        self._stop_event.set()
        if self._checkpoint_task:
            await self._checkpoint_task

        logger.info(f"Progress tracking cancelled for job {self.job_id}")

    def add_checkpoint_callback(self, callback: Callable[[Checkpoint], None]) -> None:
        """Add checkpoint creation callback"""
        self.checkpoint_callbacks.append(callback)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        with self._lock:
            return {
                'job_id': self.job_id,
                'state': self.state.value,
                'progress_percentage': self.metrics.progress_percentage,
                'success_rate': self.metrics.success_rate,
                'items_per_second': self.metrics.items_per_second,
                'elapsed_time_seconds': self.metrics.elapsed_time_seconds,
                'estimated_completion_time': self.metrics.estimated_completion_time.isoformat() if self.metrics.estimated_completion_time else None,
                'total_items': self.metrics.total_items,
                'processed_items': self.metrics.processed_items,
                'successful_items': self.metrics.successful_items,
                'failed_items': self.metrics.failed_items,
                'data_quality_score': self.metrics.data_quality_score,
                'current_position': dict(self.current_position),
                'processed_files_count': len(self.processed_files)
            }

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().isoformat()
        data = f"{self.job_id}_{timestamp}_{self.metrics.processed_items}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    async def _checkpoint_loop(self) -> None:
        """Background checkpoint creation loop"""
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(60)  # Check every minute

                # Create checkpoint if enough time has passed
                if self.state == ProgressState.RUNNING:
                    current_time = datetime.now()
                    if (self.metrics.last_update_time and
                        (current_time - self.metrics.last_update_time).total_seconds() > 300):  # 5 minutes
                        await self.create_checkpoint(CheckpointType.AUTOMATIC)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Checkpoint loop error: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        self._stop_event.set()
        if self._checkpoint_task:
            await self._checkpoint_task


# Factory function
async def create_progress_tracker(
    job_id: str,
    storage_path: Optional[str] = None,
    checkpoint_interval: int = 1000
) -> ProgressTracker:
    """Create a progress tracker with appropriate storage"""
    storage = SQLiteCheckpointStorage(storage_path or f"checkpoints_{job_id}.db")
    return ProgressTracker(job_id, storage, checkpoint_interval)