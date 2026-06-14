from __future__ import annotations

import numpy as np
import pandas as pd

from momentum_preprocessor.momentum_burst_detector import BurstConfig, annotate_dataframe, compute_features, detect_bursts


def test_annotated_cum_ret_is_measured_from_burst_start() -> None:
    closes = [100, 100, 100, 100, 100, 103, 106, 109, 112]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(closes), freq="min"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * len(closes),
        }
    )
    cfg = BurstConfig(method="pct", momentum_window=2, entry_pct=2.0, exit_pct=99.0, min_duration=2)

    featured = compute_features(df, cfg)
    bursts = detect_bursts(featured, cfg)
    annotated = annotate_dataframe(featured, bursts)

    assert len(bursts) >= 1
    start = int(bursts.iloc[0]["start_idx"])
    end = int(bursts.iloc[0]["end_idx"])
    base_close = annotated.loc[start, "close"]
    expected_pct = (annotated.loc[end, "close"] / base_close - 1) * 100

    assert annotated.loc[start, "cum_ret_pct"] == 0
    assert np.isclose(annotated.loc[end, "cum_ret_pct"], expected_pct)
    assert np.isclose(annotated.loc[end, "cum_ret"], np.log(annotated.loc[end, "close"] / base_close))
    assert annotated.loc[end, "cum_ret_pct"] != annotated.loc[end, "trigger_ret_pct"]
