"""Tests for data preprocessor module."""

from __future__ import annotations

import pandas as pd
import pytest

from data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()

    def test_process_returns_dataframe(self) -> None:
        """Test that process returns a DataFrame."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [1000.0] * 100,
        })

        result = self.preprocessor.process(df)

        assert isinstance(result, pd.DataFrame)

    def test_process_validates_columns(self) -> None:
        """Test that process validates required columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        })

        result = self.preprocessor.process(df)

        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in result.columns

    def test_process_sorts_by_timestamp(self) -> None:
        """Test that process sorts by timestamp."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-03", "2024-01-01", "2024-01-02"
            ]),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        })

        result = self.preprocessor.process(df)

        assert result.index.is_monotonic_increasing

    def test_process_handles_missing_data(self) -> None:
        """Test that process handles missing values."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "open": [100.0, None, 102.0] + [100.0] * 7,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        })

        result = self.preprocessor.process(df)

        assert len(result) > 0
