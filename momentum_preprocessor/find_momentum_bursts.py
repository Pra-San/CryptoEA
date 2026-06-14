#!/usr/bin/env python3
"""
Find variable-window momentum bursts in OHLCV CSV data.

A burst is a window where the close-to-close move is unusually large for
that exact window length, large relative to prior local volatility, and
reasonably directional rather than choppy.

Edit CONFIG below to tune sensitivity or the window range.
"""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

CONFIG = {
    # Windows are inclusive row counts: window 3 means rows s, s+1, s+2.
    "min_window_rows": 3,
    "max_window_rows": 10,

    # Candidate must be in the top 1.5% absolute close-to-close moves for
    # its own window length. Lower to 0.975 for more bursts, raise to 0.99
    # for fewer bursts.
    "window_abs_move_quantile": 0.985,

    # Absolute close-to-close move floor, in simple percent.
    "min_abs_move_pct": 0.35,

    # abs(log_return) / expected_window_volatility, where expected volatility
    # uses prior one-bar returns and sqrt(window_intervals).
    "min_momentum_score": 1.5,

    # abs(net log return) / sum(abs(one-bar log returns inside the window)).
    # 1.0 means monotonic; lower values allow more chop.
    "min_directional_efficiency": 0.60,

    # Local volatility/volume baselines use rows before the start of the window.
    "volatility_lookback_rows": 120,
    "volume_lookback_rows": 120,

    # Prevent tiny or zero local vol from inflating momentum scores too much.
    "volatility_floor_fraction_of_global": 0.35,
}

REQUIRED_COLUMNS = ["timestamp", "time", "open", "high", "low", "close", "volume"]


def stdev(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def pct(simple_ratio_minus_one):
    return simple_ratio_minus_one * 100.0


def fmt_float(value, digits=8):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.{digits}f}"
    return value


def read_rows(input_path):
    rows = []
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for i, row in enumerate(reader):
            parsed = {
                "timestamp": int(float(row["timestamp"])),
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "data_row": i + 1,
                "csv_row": i + 2,  # header is row 1 in the source CSV
                "data_index0": i,
            }
            rows.append(parsed)
    if len(rows) < CONFIG["min_window_rows"]:
        raise ValueError("Not enough rows for the configured minimum window.")
    return rows


def percentile_threshold(sorted_values, quantile):
    if not sorted_values:
        return 0.0
    q = min(max(float(quantile), 0.0), 1.0)
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def mean(values):
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def detect_bursts(rows, config):
    n = len(rows)
    min_w = int(config["min_window_rows"])
    max_w = min(int(config["max_window_rows"]), n)
    if min_w < 2:
        raise ValueError("min_window_rows must be at least 2.")
    if max_w < min_w:
        raise ValueError("max_window_rows must be >= min_window_rows.")

    closes = [r["close"] for r in rows]
    volumes = [r["volume"] for r in rows]
    log_returns = [0.0]
    for i in range(1, n):
        log_returns.append(math.log(closes[i] / closes[i - 1]))

    global_sigma = stdev(log_returns[1:])
    vol_floor = global_sigma * float(config["volatility_floor_fraction_of_global"])
    min_abs_log_move = math.log1p(float(config["min_abs_move_pct"]) / 100.0)

    abs_moves_by_window = {}
    threshold_by_window = {}
    for w in range(min_w, max_w + 1):
        values = []
        for start in range(0, n - w + 1):
            end = start + w - 1
            values.append(abs(math.log(closes[end] / closes[start])))
        values.sort()
        abs_moves_by_window[w] = values
        threshold_by_window[w] = max(
            percentile_threshold(values, config["window_abs_move_quantile"]),
            min_abs_log_move,
        )

    candidates = []
    for w in range(min_w, max_w + 1):
        abs_values_for_w = abs_moves_by_window[w]
        threshold = threshold_by_window[w]
        intervals = max(1, w - 1)
        for start in range(0, n - w + 1):
            end = start + w - 1
            start_close = closes[start]
            end_close = closes[end]
            net_log_return = math.log(end_close / start_close)
            abs_net_log_return = abs(net_log_return)
            if abs_net_log_return < threshold:
                continue

            path_abs_log_return = sum(abs(log_returns[i]) for i in range(start + 1, end + 1))
            directional_efficiency = (
                abs_net_log_return / path_abs_log_return if path_abs_log_return > 0 else 0.0
            )
            if directional_efficiency < float(config["min_directional_efficiency"]):
                continue

            prior_start = max(1, start - int(config["volatility_lookback_rows"]) + 1)
            prior_returns = log_returns[prior_start : start + 1]
            local_sigma = max(stdev(prior_returns), vol_floor)
            expected_window_vol = local_sigma * math.sqrt(intervals)
            abs_momentum_score = (
                abs_net_log_return / expected_window_vol if expected_window_vol > 0 else 0.0
            )
            if abs_momentum_score < float(config["min_momentum_score"]):
                continue

            direction_sign = 1 if net_log_return >= 0 else -1
            direction = "up" if direction_sign == 1 else "down"
            window_rows = rows[start : end + 1]
            highs = [r["high"] for r in window_rows]
            lows = [r["low"] for r in window_rows]
            window_high = max(highs)
            window_low = min(lows)
            adjusted_points = []
            for price in highs + lows:
                adjusted_points.append(direction_sign * ((price / start_close) - 1.0) * 100.0)
            max_favorable = max(0.0, max(adjusted_points))
            max_adverse = min(0.0, min(adjusted_points))

            volume_window = volumes[start : end + 1]
            prior_volume_start = max(0, start - int(config["volume_lookback_rows"]))
            prior_volumes = volumes[prior_volume_start:start]
            prior_volume_mean = mean(prior_volumes)
            volume_ratio = (
                sum(volume_window) / (prior_volume_mean * w)
                if prior_volume_mean and prior_volume_mean > 0
                else None
            )

            abs_percentile = (
                bisect.bisect_right(abs_values_for_w, abs_net_log_return) / len(abs_values_for_w) * 100.0
            )
            duration_minutes = (rows[end]["timestamp"] - rows[start]["timestamp"]) / 60000.0
            simple_move_pct = (end_close / start_close - 1.0) * 100.0
            price_change = end_close - start_close
            quality_score = abs_momentum_score * directional_efficiency

            candidates.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "direction": direction,
                    "direction_sign": direction_sign,
                    "window_rows": w,
                    "return_intervals": intervals,
                    "start_data_row": rows[start]["data_row"],
                    "end_data_row": rows[end]["data_row"],
                    "start_csv_row": rows[start]["csv_row"],
                    "end_csv_row": rows[end]["csv_row"],
                    "start_time": rows[start]["time"],
                    "end_time": rows[end]["time"],
                    "duration_minutes": duration_minutes,
                    "start_close": start_close,
                    "end_close": end_close,
                    "price_change": price_change,
                    "move_pct": simple_move_pct,
                    "abs_move_pct": abs(simple_move_pct),
                    "log_return_pct": net_log_return * 100.0,
                    "path_abs_log_return_pct": path_abs_log_return * 100.0,
                    "directional_efficiency": directional_efficiency,
                    "momentum_score": abs_momentum_score * direction_sign,
                    "abs_momentum_score": abs_momentum_score,
                    "quality_score": quality_score,
                    "local_volatility_pct_per_bar": local_sigma * 100.0,
                    "expected_window_volatility_pct": expected_window_vol * 100.0,
                    "abs_move_percentile_for_window": abs_percentile,
                    "threshold_abs_move_pct_for_window": (math.exp(threshold) - 1.0) * 100.0,
                    "window_high": window_high,
                    "window_low": window_low,
                    "intrawindow_range_pct": ((window_high - window_low) / start_close) * 100.0,
                    "max_favorable_excursion_pct": max_favorable,
                    "max_adverse_excursion_pct": max_adverse,
                    "volume_sum": sum(volume_window),
                    "volume_mean": mean(volume_window),
                    "prior_volume_mean": prior_volume_mean,
                    "volume_ratio_to_prior_mean": volume_ratio,
                    "zero_volume_bars": sum(1 for v in volume_window if v <= 0),
                    "nonzero_volume_bar_ratio": sum(1 for v in volume_window if v > 0) / w,
                    "config_min_window_rows": min_w,
                    "config_max_window_rows": max_w,
                    "config_window_abs_move_quantile": float(config["window_abs_move_quantile"]),
                    "config_min_abs_move_pct": float(config["min_abs_move_pct"]),
                    "config_min_momentum_score": float(config["min_momentum_score"]),
                    "config_min_directional_efficiency": float(config["min_directional_efficiency"]),
                }
            )

    assign_episode_ids(candidates)
    assign_quality_ranks(candidates)
    candidates.sort(key=lambda x: (x["start_data_row"], x["window_rows"], x["end_data_row"], x["direction"]))
    for i, c in enumerate(candidates, start=1):
        c["burst_id"] = f"B{i:05d}"
    return candidates, {
        "input_rows": n,
        "burst_rows": len(candidates),
        "detected_min_window_rows": min((c["window_rows"] for c in candidates), default=None),
        "detected_max_window_rows": max((c["window_rows"] for c in candidates), default=None),
        "global_volatility_pct_per_bar": global_sigma * 100.0,
        "volatility_floor_pct_per_bar": vol_floor * 100.0,
        "config": config,
    }


def assign_quality_ranks(candidates):
    ranked = sorted(candidates, key=lambda x: x["quality_score"], reverse=True)
    for rank, c in enumerate(ranked, start=1):
        c["quality_rank"] = rank


def assign_episode_ids(candidates):
    if not candidates:
        return
    group_records = []
    group_counter = 0
    for direction in ("up", "down"):
        items = [c for c in candidates if c["direction"] == direction]
        items.sort(key=lambda x: (x["start_idx"], x["end_idx"], x["window_rows"]))
        current_group = None
        current_end = None
        for c in items:
            if current_group is None or c["start_idx"] > current_end + 1:
                group_counter += 1
                current_group = f"raw_{group_counter}"
                current_end = c["end_idx"]
                group_records.append({"raw": current_group, "direction": direction, "start": c["start_idx"], "end": c["end_idx"]})
            else:
                current_end = max(current_end, c["end_idx"])
                for g in group_records:
                    if g["raw"] == current_group:
                        g["end"] = current_end
                        break
            c["episode_raw"] = current_group

    group_records.sort(key=lambda g: (g["start"], g["end"], g["direction"]))
    raw_to_final = {g["raw"]: f"E{i:04d}" for i, g in enumerate(group_records, start=1)}
    for c in candidates:
        c["episode_id"] = raw_to_final[c["episode_raw"]]
        del c["episode_raw"]


def write_bursts(candidates, output_path):
    fieldnames = [
        "burst_id",
        "episode_id",
        "quality_rank",
        "start_data_row",
        "end_data_row",
        "start_csv_row",
        "end_csv_row",
        "start_time",
        "end_time",
        "duration_minutes",
        "window_rows",
        "return_intervals",
        "direction",
        "momentum_score",
        "abs_momentum_score",
        "quality_score",
        "start_close",
        "end_close",
        "price_change",
        "move_pct",
        "abs_move_pct",
        "log_return_pct",
        "path_abs_log_return_pct",
        "directional_efficiency",
        "local_volatility_pct_per_bar",
        "expected_window_volatility_pct",
        "abs_move_percentile_for_window",
        "threshold_abs_move_pct_for_window",
        "window_high",
        "window_low",
        "intrawindow_range_pct",
        "max_favorable_excursion_pct",
        "max_adverse_excursion_pct",
        "volume_sum",
        "volume_mean",
        "prior_volume_mean",
        "volume_ratio_to_prior_mean",
        "zero_volume_bars",
        "nonzero_volume_bar_ratio",
        "config_min_window_rows",
        "config_max_window_rows",
        "config_window_abs_move_quantile",
        "config_min_abs_move_pct",
        "config_min_momentum_score",
        "config_min_directional_efficiency",
    ]
    float_digits = {
        "duration_minutes": 2,
        "momentum_score": 6,
        "abs_momentum_score": 6,
        "quality_score": 6,
        "start_close": 8,
        "end_close": 8,
        "price_change": 8,
        "move_pct": 6,
        "abs_move_pct": 6,
        "log_return_pct": 6,
        "path_abs_log_return_pct": 6,
        "directional_efficiency": 6,
        "local_volatility_pct_per_bar": 6,
        "expected_window_volatility_pct": 6,
        "abs_move_percentile_for_window": 6,
        "threshold_abs_move_pct_for_window": 6,
        "window_high": 8,
        "window_low": 8,
        "intrawindow_range_pct": 6,
        "max_favorable_excursion_pct": 6,
        "max_adverse_excursion_pct": 6,
        "volume_sum": 8,
        "volume_mean": 8,
        "prior_volume_mean": 8,
        "volume_ratio_to_prior_mean": 6,
        "nonzero_volume_bar_ratio": 6,
        "config_window_abs_move_quantile": 6,
        "config_min_abs_move_pct": 6,
        "config_min_momentum_score": 6,
        "config_min_directional_efficiency": 6,
    }
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            row = {}
            for field in fieldnames:
                value = c.get(field)
                if isinstance(value, float):
                    row[field] = fmt_float(value, float_digits.get(field, 8))
                else:
                    row[field] = value
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Find variable-window momentum bursts in OHLCV CSV data.")
    parser.add_argument("input_csv", nargs="?", default="/mnt/data/temp.csv")
    parser.add_argument("output_csv", nargs="?", default="/mnt/data/momentum_bursts.csv")
    parser.add_argument("--summary-json", default="/mnt/data/momentum_burst_summary.json")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    candidates, summary = detect_bursts(rows, CONFIG)
    write_bursts(candidates, args.output_csv)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
