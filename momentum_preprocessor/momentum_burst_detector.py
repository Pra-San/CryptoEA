"""
momentum_burst_detector.py

Template for finding "momentum bursts" in 1-minute OHLCV data (Binance-style
CSVs: timestamp, time, open, high, low, close, volume) and marking their
start / end bars so you can study what happens before/during/after them.

------------------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------------------
1. For every bar, compute the trigger log-return over the last
   `momentum_window` bars (i.e. "how far did price move in the last N
   minutes").
2. Compare that move to the asset's recent "normal" volatility
   (rolling std-dev of 1-bar returns over `vol_window` bars) to get a
   z-score. A z-score of 3 means "this N-minute move is ~3x bigger than
   a typical N-minute move for this asset right now". Using a z-score
   instead of a fixed % keeps the detector comparable across BTC/ETH/SOL
   even though their typical volatility differs a lot.
3. A burst STARTS the first bar where |z| crosses above `entry_z`.
4. The burst is considered ongoing until |z| drops back below `exit_z`
   (a lower threshold than entry -> hysteresis, so the burst doesn't
   flicker on/off bar by bar).
5. Bursts shorter than `min_duration` bars are dropped as noise.
6. Optional: require a volume spike (volume z-score) to confirm.

There's also a simpler `method="pct"` mode that skips the volatility
normalization and just uses a flat % move threshold, if you'd rather
tune in plain percentage terms.

------------------------------------------------------------------------
WHAT YOU GET
------------------------------------------------------------------------
- `bursts_df`     : one row per detected burst (start/end index & time,
                    direction, duration, % move, peak z-score)
- `annotated_df`  : your original df + columns `in_burst`, `burst_id`,
                    `burst_direction`, plus the computed features. In the
                    annotated output, `cum_ret` means return from that
                    burst's start bar to the current bar.
- `build_event_panel(...)` : stacks N bars before/after every burst
                    start (or end) into one DataFrame so you can average
                    price/volume paths across bursts and look for
                    repeatable patterns (e.g. "does volume spike 3 bars
                    BEFORE the burst actually starts?")

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
    python momentum_burst_detector.py /path/to/btcusdt_1m_spot.csv

or import the pieces directly:

    from momentum_burst_detector import BurstConfig, load_ohlcv, \
        compute_features, detect_bursts, annotate_dataframe, build_event_panel

    cfg = BurstConfig(momentum_window=5, vol_window=60, entry_z=3.0)
    df = load_ohlcv("btcusdt_1m_spot.csv")
    df = compute_features(df, cfg)
    bursts = detect_bursts(df, cfg)
    df = annotate_dataframe(df, bursts)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG  -- this is the main thing you'll want to tune
# ============================================================================

@dataclass
class BurstConfig:
    # "zscore" (volatility-normalized, recommended for comparing assets)
    # or "pct" (flat percentage move threshold)
    method: str = "zscore"

    # How many bars define the "move" you're checking (the burst window).
    momentum_window: int = 5

    # How many bars of history define "normal" volatility (zscore method).
    vol_window: int = 60

    # zscore method thresholds
    entry_z: float = 3.0    # |z| above this STARTS a burst
    exit_z: float = 1.0     # |z| below this ENDS a burst (hysteresis)

    # pct method thresholds (in percent, e.g. 0.5 = 0.5%)
    entry_pct: float = 0.5
    exit_pct: float = 0.15

    # Drop bursts shorter than this many bars (noise filter).
    min_duration: int = 2

    # Bars to wait after a burst ends before a new one can start.
    # 0 = no cooldown, allow immediate re-trigger.
    cooldown: int = 0

    # If |1-bar return| falls below this during an active burst, end the
    # burst immediately even if the rolling z-score is still above exit_z.
    # This prevents the rolling window from dragging out the burst with
    # old momentum while price has already flatlined.  0 = disabled.
    fade_ret_threshold: float = 0.0001

    # Optional volume confirmation: require volume z-score above
    # `volume_z_threshold` on the bar that triggers the burst.
    require_volume: bool = False
    volume_window: int = 60
    volume_z_threshold: float = 1.5


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load a Binance-style 1m CSV (timestamp, time, open, high, low, close, volume)."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_features(df: pd.DataFrame, config: BurstConfig) -> pd.DataFrame:
    """Add the columns the detector needs (and a few extras useful for
    pattern analysis later: ATR, RSI)."""
    df = df.copy()

    # --- core momentum / volatility ---
    df["ret"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling move used only to trigger/score a burst. This is intentionally
    # fixed-width; burst-relative `cum_ret` is filled in annotate_dataframe().
    df["trigger_ret"] = df["ret"].rolling(config.momentum_window).sum()
    df["trigger_ret_pct"] = (np.exp(df["trigger_ret"]) - 1) * 100

    vol = df["ret"].rolling(config.vol_window).std()
    vol = vol.replace(0, np.nan)
    df["vol"] = vol
    df["z"] = df["trigger_ret"] / (df["vol"] * np.sqrt(config.momentum_window))

    # Recent 1-bar return (used by fade detection to end bursts when price
    # flatlines, even if the rolling window still contains old momentum).
    df["recent_ret"] = df["ret"]

    # These are burst-relative outputs, not fixed-window trigger features.
    # They are NaN until annotate_dataframe() knows each burst's start bar.
    df["cum_ret"] = np.nan
    df["cum_ret_pct"] = np.nan

    # --- optional volume z-score ---
    vol_mean = df["volume"].rolling(config.volume_window).mean()
    vol_std = df["volume"].rolling(config.volume_window).std().replace(0, np.nan)
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std

    # --- extras that are often useful when studying what happens
    #     around a burst (feel free to add more here) ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


# ============================================================================
# BURST DETECTION
# ============================================================================

def _score_and_thresholds(row: pd.Series, config: BurstConfig) -> float:
    if config.method == "zscore":
        return row["z"]
    elif config.method == "pct":
        return row["trigger_ret_pct"]
    else:
        raise ValueError(f"Unknown method: {config.method}")


def detect_bursts(df: pd.DataFrame, config: BurstConfig) -> pd.DataFrame:
    """Run the start/ongoing/end state machine over the whole series.

    Returns a DataFrame with one row per burst:
        start_idx, end_idx, start_time, end_time, direction,
        duration_bars, price_start, price_end, pct_move, peak_score
    """
    if config.method == "zscore":
        score_col = "z"
        entry_thr, exit_thr = config.entry_z, config.exit_z
    elif config.method == "pct":
        score_col = "trigger_ret_pct"
        entry_thr, exit_thr = config.entry_pct, config.exit_pct
    else:
        raise ValueError(f"Unknown method: {config.method}")

    scores = df[score_col].to_numpy()
    volume_z = df["volume_z"].to_numpy() if config.require_volume else None
    recent_ret = df["recent_ret"].to_numpy()
    fade_thr = config.fade_ret_threshold

    bursts = []
    state = 0          # 0 = none, 1 = up-burst, -1 = down-burst
    start_idx = None
    peak_score = 0.0
    cooldown_left = 0

    n = len(df)
    for i in range(n):
        s = scores[i]
        if np.isnan(s):
            continue

        if state == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue

            triggered_dir = 0
            if s > entry_thr:
                triggered_dir = 1
            elif s < -entry_thr:
                triggered_dir = -1

            if triggered_dir != 0:
                if config.require_volume:
                    vz = volume_z[i]
                    if np.isnan(vz) or vz < config.volume_z_threshold:
                        continue
                state = triggered_dir
                start_idx = i
                peak_score = s

        elif state == 1:
            peak_score = max(peak_score, s)
            # End burst early if price has flatlined (momentum has faded)
            # even though the rolling window still contains old momentum.
            if fade_thr > 0 and abs(recent_ret[i]) < fade_thr:
                bursts.append(_build_burst(df, start_idx, i, 1, peak_score))
                state = 0
                cooldown_left = config.cooldown
            elif s < exit_thr:
                bursts.append(_build_burst(df, start_idx, i, 1, peak_score))
                state = 0
                cooldown_left = config.cooldown

        elif state == -1:
            peak_score = min(peak_score, s)
            # End burst early if price has flatlined (momentum has faded)
            # even though the rolling window still contains old momentum.
            if fade_thr > 0 and abs(recent_ret[i]) < fade_thr:
                bursts.append(_build_burst(df, start_idx, i, -1, peak_score))
                state = 0
                cooldown_left = config.cooldown
            elif s > -exit_thr:
                bursts.append(_build_burst(df, start_idx, i, -1, peak_score))
                state = 0
                cooldown_left = config.cooldown

    # close out a burst still running at the end of the data
    if state != 0 and start_idx is not None:
        bursts.append(_build_burst(df, start_idx, n - 1, state, peak_score))

    bursts_df = pd.DataFrame(bursts)
    if bursts_df.empty:
        return bursts_df

    bursts_df = bursts_df[bursts_df["duration_bars"] >= config.min_duration]
    return bursts_df.reset_index(drop=True)


def _build_burst(df, start_idx, end_idx, direction, peak_score) -> dict:
    price_start = df["close"].iloc[start_idx]
    price_end = df["close"].iloc[end_idx]
    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": df["time"].iloc[start_idx],
        "end_time": df["time"].iloc[end_idx],
        "direction": "up" if direction == 1 else "down",
        "duration_bars": end_idx - start_idx + 1,
        "price_start": price_start,
        "price_end": price_end,
        "pct_move": (price_end / price_start - 1) * 100,
        "peak_score": peak_score,
    }


# ============================================================================
# ANNOTATION
# ============================================================================

def annotate_dataframe(df: pd.DataFrame, bursts_df: pd.DataFrame) -> pd.DataFrame:
    """Add `in_burst`, `burst_id`, `burst_direction`, and `burst_phase`
    (start / middle / end) columns to the original dataframe.

    Also fills `cum_ret` / `cum_ret_pct` as burst-relative returns: each
    annotated bar is measured from its own burst's start close to the current
    close, not from the fixed trigger window.
    """
    df = df.copy()
    df["in_burst"] = False
    df["burst_id"] = -1
    df["burst_direction"] = ""
    df["burst_phase"] = ""
    df["cum_ret"] = np.nan
    df["cum_ret_pct"] = np.nan

    for i, row in bursts_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        mask = (df.index >= s) & (df.index <= e)
        df.loc[mask, "in_burst"] = True
        df.loc[mask, "burst_id"] = i
        df.loc[mask, "burst_direction"] = row["direction"]
        df.loc[s, "burst_phase"] = "start"
        df.loc[e, "burst_phase"] = "end"
        base_close = float(df.loc[s, "close"])
        if base_close > 0:
            burst_close = df.loc[mask, "close"].astype(float)
            df.loc[mask, "cum_ret"] = np.log(burst_close / base_close)
            df.loc[mask, "cum_ret_pct"] = (burst_close / base_close - 1) * 100
        if e > s:
            df.loc[(df.index > s) & (df.index < e), "burst_phase"] = "middle"

    return df


# ============================================================================
# PATTERN / EVENT WINDOW ANALYSIS
# ============================================================================

DEFAULT_PANEL_COLUMNS = ["open", "high", "low", "close", "volume", "ret", "volume_z", "atr", "rsi"]


def extract_event_window(
    df: pd.DataFrame,
    center_idx: int,
    pre: int = 30,
    post: int = 30,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Slice `pre` bars before and `post` bars after `center_idx`.
    Adds a `rel_idx` column: -pre ... 0 ... +post, where 0 = center_idx."""
    if columns is None:
        columns = DEFAULT_PANEL_COLUMNS

    lo = max(0, center_idx - pre)
    hi = min(len(df) - 1, center_idx + post)
    window = df.iloc[lo:hi + 1].copy()
    window["rel_idx"] = window.index - center_idx
    keep = ["rel_idx"] + [c for c in columns if c in window.columns]
    return window[keep]


def build_event_panel(
    df: pd.DataFrame,
    bursts_df: pd.DataFrame,
    event: str = "start",
    pre: int = 30,
    post: int = 30,
    columns: list[str] | None = None,
    normalize_price: bool = True,
) -> pd.DataFrame:
    """Stack the windows around every burst's start (or end) bar into one
    long DataFrame, tagged with `burst_id` and `direction`.

    If `normalize_price=True`, open/high/low/close are converted to percent
    change relative to the price at the event bar, so paths from different
    bursts (and different price levels) are directly comparable / averageable.

    Typical next step:
        panel = build_event_panel(df, bursts, event="start", pre=20, post=20)
        avg_path = panel.groupby(["direction", "rel_idx"])["close"].mean()
    """
    if event not in ("start", "end"):
        raise ValueError("event must be 'start' or 'end'")
    if columns is None:
        columns = DEFAULT_PANEL_COLUMNS

    panels = []
    for i, row in bursts_df.iterrows():
        center_idx = row["start_idx"] if event == "start" else row["end_idx"]
        w = extract_event_window(df, center_idx, pre, post, columns)

        if normalize_price:
            base_price = df["close"].iloc[center_idx]
            for col in ("open", "high", "low", "close"):
                if col in w.columns:
                    w[col] = (w[col] / base_price - 1) * 100  # % change vs event bar

        # Because bursts have different durations, the same rel_idx can fall
        # "during" one burst and "after" another. Tag each bar so you can
        # filter/split on this before averaging across bursts.
        abs_idx = center_idx + w["rel_idx"]
        w["in_this_burst"] = (abs_idx >= row["start_idx"]) & (abs_idx <= row["end_idx"])

        w["burst_id"] = i
        w["direction"] = row["direction"]
        w["duration_bars"] = row["duration_bars"]
        panels.append(w)

    if not panels:
        return pd.DataFrame()
    return pd.concat(panels, ignore_index=True)


def build_normalized_burst_shape(
    df: pd.DataFrame,
    bursts_df: pd.DataFrame,
    n_points: int = 20,
    columns: list[str] | None = None,
    normalize_price: bool = True,
) -> pd.DataFrame:
    """Resample the INTERIOR of each burst (start_idx..end_idx inclusive) onto
    a common 0->1 "progress" axis with `n_points` points, using linear
    interpolation.

    This sidesteps the variable-duration problem in `build_event_panel`:
    a 3-bar burst and a 16-bar burst both get mapped to the same
    progress=0.0 (start) ... progress=1.0 (end) scale, so you can average
    "shape" across bursts regardless of how long each one actually took.

    Typical next step:
        shape = build_normalized_burst_shape(df, bursts)
        avg_shape = shape.groupby(["direction", "progress"])["close"].mean()
    """
    if columns is None:
        columns = ["close", "volume", "volume_z", "atr", "rsi"]

    panels = []
    for i, row in bursts_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        seg = df.iloc[s:e + 1].reset_index(drop=True)
        if len(seg) < 2:
            seg = pd.concat([seg, seg], ignore_index=True)  # need >=2 points to interpolate

        x_old = np.linspace(0.0, 1.0, len(seg))
        x_new = np.linspace(0.0, 1.0, n_points)

        resampled = {"progress": x_new}
        for col in columns:
            if col in seg.columns:
                resampled[col] = np.interp(x_new, x_old, seg[col].to_numpy())

        w = pd.DataFrame(resampled)
        if normalize_price and "close" in w.columns:
            base_price = seg["close"].iloc[0]
            w["close"] = (w["close"] / base_price - 1) * 100  # % change vs burst start

        w["burst_id"] = i
        w["direction"] = row["direction"]
        w["duration_bars"] = row["duration_bars"]
        panels.append(w)

    if not panels:
        return pd.DataFrame()
    return pd.concat(panels, ignore_index=True)


# ============================================================================
# SUMMARY / PLOTTING HELPERS
# ============================================================================

def summarize_bursts(bursts_df: pd.DataFrame) -> None:
    if bursts_df.empty:
        print("No bursts detected with the current config.")
        return

    print(f"Total bursts: {len(bursts_df)}")
    print("\nBy direction:")
    print(bursts_df["direction"].value_counts())
    print("\nDuration (bars) & % move stats:")
    print(bursts_df[["duration_bars", "pct_move"]].describe())


def plot_bursts(df: pd.DataFrame, bursts_df: pd.DataFrame, start: int = 0, end: int | None = None):
    """Quick visual sanity check: price line with shaded burst regions
    (green = up burst, red = down burst). Requires matplotlib."""
    import matplotlib.pyplot as plt

    if end is None:
        end = len(df)

    fig, ax = plt.subplots(figsize=(14, 6))
    sub = df.iloc[start:end]
    ax.plot(sub["time"], sub["close"], color="black", linewidth=0.8)

    for _, row in bursts_df.iterrows():
        if row["end_idx"] < start or row["start_idx"] > end:
            continue
        color = "green" if row["direction"] == "up" else "red"
        ax.axvspan(
            df["time"].iloc[row["start_idx"]],
            df["time"].iloc[row["end_idx"]],
            color=color,
            alpha=0.2,
        )

    ax.set_title("Price with momentum bursts highlighted (green=up, red=down)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    fig.tight_layout()
    return fig, ax


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect momentum bursts in 1m OHLCV data.")
    parser.add_argument("csv_path", help="Path to a Binance-style 1m OHLCV CSV")
    parser.add_argument("--method", choices=["zscore", "pct"], default="zscore")
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--entry-z", type=float, default=3.0)
    parser.add_argument("--exit-z", type=float, default=1.0)
    parser.add_argument("--entry-pct", type=float, default=0.5)
    parser.add_argument("--exit-pct", type=float, default=0.15)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument("--require-volume", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of bursts")
    parser.add_argument("--out-prefix", default="momentum_bursts", help="Prefix for output CSVs")
    args = parser.parse_args(argv)

    config = BurstConfig(
        method=args.method,
        momentum_window=args.momentum_window,
        vol_window=args.vol_window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        entry_pct=args.entry_pct,
        exit_pct=args.exit_pct,
        min_duration=args.min_duration,
        cooldown=args.cooldown,
        require_volume=args.require_volume,
    )

    df = load_ohlcv(args.csv_path)
    df = compute_features(df, config)
    bursts = detect_bursts(df, config)
    summarize_bursts(bursts)

    annotated = annotate_dataframe(df, bursts)
    annotated_path = f"{args.out_prefix}_annotated.csv"
    bursts_path = f"{args.out_prefix}_bursts.csv"
    annotated.to_csv(annotated_path, index=False)
    bursts.to_csv(bursts_path, index=False)
    print(f"\nWrote {annotated_path} ({len(annotated)} rows)")
    print(f"Wrote {bursts_path} ({len(bursts)} bursts)")

    if args.plot and not bursts.empty:
        fig, _ = plot_bursts(df, bursts)
        plot_path = f"{args.out_prefix}_plot.png"
        fig.savefig(plot_path, dpi=120)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
