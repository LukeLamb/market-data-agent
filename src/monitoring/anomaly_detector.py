"""
Machine Learning Anomaly Detection for Market Data
Phase 4 Step 2: Enterprise Monitoring & Observability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime, timedelta
import redis
from .metrics_exporter import metrics_exporter


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    LATENCY_ANOMALY = "latency_anomaly"
    ERROR_RATE_ANOMALY = "error_rate_anomaly"
    DATA_QUALITY_ANOMALY = "data_quality_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    symbol: str
    timestamp: datetime
    anomaly_type: AnomalyType
    score: float
    threshold: float
    is_anomaly: bool
    confidence: float
    features: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseAnomalyDetector:
    """Base class for anomaly detectors"""

    def __init__(self, name: str, threshold: float = 0.1):
        self.name = name
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.trained = False

    def fit(self, data: pd.DataFrame) -> None:
        """Train the anomaly detection model"""
        raise NotImplementedError

    def predict(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies in the data"""
        raise NotImplementedError

    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'threshold': self.threshold,
                'trained': self.trained
            }, path)

    def load_model(self, path: str) -> None:
        """Load a pre-trained model"""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.threshold = data['threshold']
            self.trained = data['trained']
            self.logger.info(f"Loaded model from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {e}")


class PriceAnomalyDetector(BaseAnomalyDetector):
    """Detect price anomalies using isolation forest"""

    def __init__(self, contamination: float = 0.1):
        super().__init__("price_anomaly", threshold=0.1)
        self.contamination = contamination

    def fit(self, data: pd.DataFrame) -> None:
        """Train on historical price data"""
        try:
            # Extract features for price anomaly detection
            features = self._extract_features(data)

            if len(features) < 10:
                self.logger.warning("Insufficient data for training")
                return

            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Train isolation forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(scaled_features)
            self.trained = True

            self.logger.info(f"Trained price anomaly detector on {len(features)} samples")

        except Exception as e:
            self.logger.error(f"Failed to train price anomaly detector: {e}")

    def predict(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect price anomalies"""
        if not self.trained:
            return []

        try:
            features = self._extract_features(data)
            if len(features) == 0:
                return []

            scaled_features = self.scaler.transform(features)

            # Get anomaly scores
            scores = self.model.decision_function(scaled_features)
            predictions = self.model.predict(scaled_features)

            results = []
            for i, (score, pred) in enumerate(zip(scores, predictions)):
                # Convert isolation forest score to 0-1 range
                normalized_score = max(0, min(1, (1 - score) / 2))

                if pred == -1:  # Anomaly detected
                    result = AnomalyResult(
                        symbol=data.iloc[i]['symbol'],
                        timestamp=data.iloc[i]['timestamp'],
                        anomaly_type=AnomalyType.PRICE_SPIKE,
                        score=normalized_score,
                        threshold=self.threshold,
                        is_anomaly=True,
                        confidence=min(1.0, normalized_score * 2),
                        features=features.iloc[i].to_dict(),
                        metadata={'isolation_forest_score': score}
                    )
                    results.append(result)

                    # Update metrics
                    metrics_exporter.update_anomaly_score(
                        data.iloc[i]['symbol'],
                        AnomalyType.PRICE_SPIKE.value,
                        normalized_score
                    )

            return results

        except Exception as e:
            self.logger.error(f"Failed to detect price anomalies: {e}")
            return []

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for price anomaly detection"""
        features = []

        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')

            if len(symbol_data) < 5:
                continue

            # Calculate price-based features
            symbol_data['price_change'] = symbol_data['price'].pct_change()
            symbol_data['price_ma_5'] = symbol_data['price'].rolling(5).mean()
            symbol_data['price_std_5'] = symbol_data['price'].rolling(5).std()
            symbol_data['volume_change'] = symbol_data['volume'].pct_change()

            # Z-score of price change
            symbol_data['price_zscore'] = (
                symbol_data['price_change'] - symbol_data['price_change'].mean()
            ) / symbol_data['price_change'].std()

            # Volatility features
            symbol_data['volatility'] = symbol_data['price_change'].rolling(10).std()
            symbol_data['volume_volatility'] = symbol_data['volume_change'].rolling(10).std()

            # Remove NaN values and select features
            feature_cols = [
                'price_change', 'price_zscore', 'volatility',
                'volume_change', 'volume_volatility'
            ]

            symbol_features = symbol_data[feature_cols].dropna()
            features.append(symbol_features)

        if features:
            return pd.concat(features, ignore_index=True)
        else:
            return pd.DataFrame()


class SystemAnomalyDetector(BaseAnomalyDetector):
    """Detect system performance anomalies"""

    def __init__(self):
        super().__init__("system_anomaly", threshold=0.15)

    def fit(self, data: pd.DataFrame) -> None:
        """Train on system metrics"""
        try:
            features = self._extract_system_features(data)

            if len(features) < 20:
                self.logger.warning("Insufficient system data for training")
                return

            scaled_features = self.scaler.fit_transform(features)

            self.model = IsolationForest(
                contamination=0.05,  # Expect fewer system anomalies
                random_state=42,
                n_estimators=100
            )
            self.model.fit(scaled_features)
            self.trained = True

            self.logger.info(f"Trained system anomaly detector on {len(features)} samples")

        except Exception as e:
            self.logger.error(f"Failed to train system anomaly detector: {e}")

    def predict(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect system anomalies"""
        if not self.trained:
            return []

        try:
            features = self._extract_system_features(data)
            if len(features) == 0:
                return []

            scaled_features = self.scaler.transform(features)
            scores = self.model.decision_function(scaled_features)
            predictions = self.model.predict(scaled_features)

            results = []
            for i, (score, pred) in enumerate(zip(scores, predictions)):
                normalized_score = max(0, min(1, (1 - score) / 2))

                if pred == -1:
                    # Determine anomaly type based on features
                    feature_row = features.iloc[i]
                    anomaly_type = self._classify_system_anomaly(feature_row)

                    result = AnomalyResult(
                        symbol="SYSTEM",
                        timestamp=data.iloc[i]['timestamp'],
                        anomaly_type=anomaly_type,
                        score=normalized_score,
                        threshold=self.threshold,
                        is_anomaly=True,
                        confidence=min(1.0, normalized_score * 2),
                        features=feature_row.to_dict(),
                        metadata={'isolation_forest_score': score}
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Failed to detect system anomalies: {e}")
            return []

    def _extract_system_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract system performance features"""
        feature_cols = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'response_time', 'error_rate', 'request_rate'
        ]

        # Fill missing values and calculate rolling statistics
        features = data[feature_cols].fillna(0)

        # Add rolling averages and standard deviations
        for col in feature_cols:
            features[f'{col}_ma_10'] = features[col].rolling(10).mean()
            features[f'{col}_std_10'] = features[col].rolling(10).std()

        return features.dropna()

    def _classify_system_anomaly(self, features: pd.Series) -> AnomalyType:
        """Classify the type of system anomaly"""
        if features['response_time'] > features['response_time_ma_10'] * 2:
            return AnomalyType.LATENCY_ANOMALY
        elif features['error_rate'] > features['error_rate_ma_10'] * 3:
            return AnomalyType.ERROR_RATE_ANOMALY
        else:
            return AnomalyType.PATTERN_ANOMALY


class AnomalyDetectionEngine:
    """Main anomaly detection engine"""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.logger = logging.getLogger(__name__)
        self.detectors = {
            'price': PriceAnomalyDetector(),
            'system': SystemAnomalyDetector()
        }
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.running = False

    async def start(self):
        """Start the anomaly detection engine"""
        self.running = True
        self.logger.info("Starting anomaly detection engine")

        # Load pre-trained models if available
        await self._load_models()

        # Start detection loop
        asyncio.create_task(self._detection_loop())

    async def stop(self):
        """Stop the anomaly detection engine"""
        self.running = False
        self.logger.info("Stopping anomaly detection engine")

    async def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """Train anomaly detection models"""
        self.logger.info("Training anomaly detection models")

        for detector_name, detector in self.detectors.items():
            if detector_name in training_data:
                self.logger.info(f"Training {detector_name} detector")
                detector.fit(training_data[detector_name])

                # Save trained model
                model_path = f"/tmp/{detector_name}_anomaly_model.pkl"
                detector.save_model(model_path)

    async def detect_anomalies(self, data: Dict[str, pd.DataFrame]) -> List[AnomalyResult]:
        """Detect anomalies in real-time data"""
        all_results = []

        for detector_name, detector in self.detectors.items():
            if detector_name in data and detector.trained:
                results = detector.predict(data[detector_name])
                all_results.extend(results)

        # Store results in Redis for alerting
        await self._store_anomaly_results(all_results)

        return all_results

    async def _detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                # Get recent data from Redis/database
                data = await self._get_recent_data()

                if data:
                    anomalies = await self.detect_anomalies(data)

                    if anomalies:
                        self.logger.info(f"Detected {len(anomalies)} anomalies")
                        await self._process_anomalies(anomalies)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(60)

    async def _load_models(self):
        """Load pre-trained models"""
        for detector_name, detector in self.detectors.items():
            model_path = f"/tmp/{detector_name}_anomaly_model.pkl"
            try:
                detector.load_model(model_path)
            except FileNotFoundError:
                self.logger.info(f"No pre-trained model found for {detector_name}")

    async def _get_recent_data(self) -> Dict[str, pd.DataFrame]:
        """Get recent data for anomaly detection"""
        # This would typically fetch from your time series database
        # For now, return empty dict as placeholder
        return {}

    async def _store_anomaly_results(self, results: List[AnomalyResult]):
        """Store anomaly results in Redis"""
        for result in results:
            key = f"anomaly:{result.symbol}:{result.timestamp.isoformat()}"
            value = {
                'anomaly_type': result.anomaly_type.value,
                'score': result.score,
                'confidence': result.confidence,
                'is_anomaly': result.is_anomaly
            }

            # Store with 24 hour expiry
            self.redis_client.setex(key, 86400, json.dumps(value))

    async def _process_anomalies(self, anomalies: List[AnomalyResult]):
        """Process detected anomalies (alerting, etc.)"""
        for anomaly in anomalies:
            if anomaly.score > 0.8:  # High severity
                self.logger.warning(
                    f"High severity anomaly detected: {anomaly.symbol} "
                    f"({anomaly.anomaly_type.value}) - Score: {anomaly.score:.3f}"
                )

                # Trigger alert via metrics
                metrics_exporter.update_anomaly_score(
                    anomaly.symbol,
                    anomaly.anomaly_type.value,
                    anomaly.score
                )


# Global anomaly detection engine instance
anomaly_engine = AnomalyDetectionEngine()


async def initialize_anomaly_detection():
    """Initialize and start anomaly detection"""
    await anomaly_engine.start()
    return anomaly_engine


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        engine = await initialize_anomaly_detection()

        # In a real implementation, you would:
        # 1. Load historical data for training
        # 2. Train the models
        # 3. Start real-time detection

        print("Anomaly detection engine started")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await engine.stop()

    asyncio.run(main())