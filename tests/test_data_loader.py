"""Tests for data loader module."""

from __future__ import annotations

import pandas as pd
import pytest

from data.loader import load_csv, load_manifest


class TestLoadCsv:
    """Tests for load_csv function."""

    def test_load_csv_returns_dataframe(self, tmp_path) -> None:
        """Test that load_csv returns a DataFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01 00:00:00,100,101,99,100.5,1000\n"
            "2024-01-01 01:00:00,100.5,102,100,101,1200\n"
        )

        df = load_csv(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_csv_sets_timestamp_index(self, tmp_path) -> None:
        """Test that timestamp is set as index."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01 00:00:00,100,101,99,100.5,1000\n"
        )

        df = load_csv(str(csv_file))

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_csv_filters_by_date(self, tmp_path) -> None:
        """Test date filtering."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01 00:00:00,100,101,99,100.5,1000\n"
            "2024-02-01 00:00:00,101,102,100,101,1200\n"
            "2024-03-01 00:00:00,102,103,101,102,1300\n"
        )

        df = load_csv(
            str(csv_file),
            start_date="2024-02-01",
            end_date="2024-02-28",
        )

        assert len(df) == 1

    def test_load_csv_empty_file(self, tmp_path) -> None:
        """Test handling of empty CSV files."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        df = load_csv(str(csv_file))

        assert df.empty


class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_load_manifest_returns_dict(self, tmp_path) -> None:
        """Test that load_manifest returns a dictionary."""
        import json

        manifest_file = tmp_path / "manifest.json"
        manifest_data = {
            "symbols": ["btcusdt", "ethusdt"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        }

        manifest_file.write_text(json.dumps(manifest_data))

        manifest = load_manifest(str(manifest_file))

        assert isinstance(manifest, dict)
        assert manifest["symbols"] == ["btcusdt", "ethusdt"]
