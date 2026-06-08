#!/usr/bin/env python3
"""Build a static dashboard for optimization and walk-forward research runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent
OPT_ROOT = PROJECT_ROOT / "research" / "optimization"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_best_value(best: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in best:
            return best[name]
    return None


def summarize_best(path: Path) -> dict[str, Any]:
    data = load_json(path)
    best = data.get("best", {})
    costs = data.get("costs", {})
    run_dir = path.parent
    return {
        "run_dir": run_dir.name,
        "kind": run_dir.name.rsplit("_", 3)[0],
        "summary_file": str(path.relative_to(OPT_ROOT)),
        "symbol": data.get("symbol"),
        "timeframe": data.get("timeframe"),
        "family": best.get("family", "v3"),
        "trial": best.get("trial"),
        "trials": data.get("trials"),
        "seed": data.get("seed"),
        "fee_rate": costs.get("fee_rate"),
        "slippage_rate": costs.get("slippage_rate"),
        "stress_slippage_rate": costs.get("stress_slippage_rate"),
        "deploy_score": best.get("deploy_score"),
        "wf_score": best.get("wf_score"),
        "wf_positive_exp": best.get("wf_positive_exp"),
        "wf_profitable": best.get("wf_profitable"),
        "wf_windows": best.get("wf_windows"),
        "wf_mean_exp_r": best.get("wf_mean_exp_r"),
        "wf_min_exp_r": best.get("wf_min_exp_r"),
        "wf_max_dd_r": best.get("wf_max_dd_r"),
        "wf_total_trades": best.get("wf_total_trades"),
        "validation_trades": get_best_value(best, "validation_total_trades"),
        "validation_pf": get_best_value(best, "validation_profit_factor"),
        "validation_exp_r": get_best_value(best, "validation_expectancy_r"),
        "validation_dd_r": get_best_value(best, "validation_max_drawdown_r"),
        "validation_dd_pct": get_best_value(best, "validation_max_drawdown_pct"),
        "validation_return_pct": get_best_value(best, "validation_total_return_pct"),
        "stress_trades": get_best_value(best, "stress_total_trades"),
        "stress_pf": get_best_value(best, "stress_profit_factor"),
        "stress_exp_r": get_best_value(best, "stress_expectancy_r"),
        "stress_dd_r": get_best_value(best, "stress_max_drawdown_r"),
        "stress_dd_pct": get_best_value(best, "stress_max_drawdown_pct"),
        "stress_return_pct": get_best_value(best, "stress_total_return_pct"),
        "best_params": json.dumps(
            {key: best[key] for key in sorted(best) if not key.startswith(("train_", "validation_", "stress_", "wf_"))},
            sort_keys=True,
        ),
    }


def summarize_walk_forward(path: Path) -> dict[str, Any]:
    data = load_json(path)
    aggregate = data.get("aggregate", {})
    costs = data.get("costs", {})
    portfolio_metrics = data.get("portfolio_metrics", {})
    is_portfolio = bool(portfolio_metrics)
    run_dir = path.parent
    return {
        "run_dir": run_dir.name,
        "kind": run_dir.name.rsplit("_", 3)[0],
        "summary_file": str(path.relative_to(OPT_ROOT)),
        "symbol": data.get("symbol"),
        "timeframe": data.get("timeframe"),
        "family": "portfolio" if is_portfolio else data.get("params", {}).get("family", "v3"),
        "trial": None,
        "trials": None,
        "seed": None,
        "fee_rate": costs.get("fee_rate"),
        "slippage_rate": costs.get("slippage_rate"),
        "stress_slippage_rate": None,
        "deploy_score": None,
        "wf_score": None,
        "wf_positive_exp": aggregate.get("positive_expectancy_windows"),
        "wf_profitable": aggregate.get("profitable_windows"),
        "wf_windows": aggregate.get("windows"),
        "wf_mean_exp_r": aggregate.get("mean_expectancy_r"),
        "wf_min_exp_r": aggregate.get("min_expectancy_r"),
        "wf_max_dd_r": aggregate.get("max_drawdown_r"),
        "wf_total_trades": aggregate.get("total_trades"),
        "validation_trades": portfolio_metrics.get("total_trades"),
        "validation_pf": portfolio_metrics.get("profit_factor"),
        "validation_exp_r": portfolio_metrics.get("expectancy_r"),
        "validation_dd_r": portfolio_metrics.get("max_drawdown_r"),
        "validation_dd_pct": portfolio_metrics.get("max_drawdown_pct"),
        "validation_return_pct": portfolio_metrics.get("total_return_pct"),
        "stress_trades": None,
        "stress_pf": None,
        "stress_exp_r": None,
        "stress_dd_r": None,
        "stress_dd_pct": None,
        "stress_return_pct": None,
        "best_params": json.dumps(data.get("params", data.get("candidates", {})), sort_keys=True),
    }


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OPT_ROOT.glob("*/best_summary.json")):
        rows.append(summarize_best(path))
    for path in sorted(OPT_ROOT.glob("*/summary.json")):
        rows.append(summarize_walk_forward(path))
    rows.sort(key=lambda row: row["run_dir"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def build_html(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        summary = row["summary_file"]
        table_rows.append(
            "<tr>"
            f"<td><a href='{summary}'>{row['run_dir']}</a></td>"
            f"<td>{fmt(row['symbol'])}</td>"
            f"<td>{fmt(row['timeframe'])}</td>"
            f"<td>{fmt(row['family'])}</td>"
            f"<td>{fmt(row['stress_trades'] or row['validation_trades'] or row['wf_total_trades'])}</td>"
            f"<td>{fmt(row['stress_pf'] or row['validation_pf'])}</td>"
            f"<td>{fmt(row['stress_exp_r'] or row['validation_exp_r'] or row['wf_mean_exp_r'])}</td>"
            f"<td>{fmt(row['stress_dd_r'] or row['validation_dd_r'] or row['wf_max_dd_r'])}</td>"
            f"<td>{fmt(row['wf_positive_exp'])}/{fmt(row['wf_windows'])}</td>"
            f"<td>{fmt(row['deploy_score'] or row['wf_score'])}</td>"
            f"<td><code>{fmt(row['best_params'])}</code></td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CryptoEA Optimization Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #667085;
      --line: #d7dde5;
      --accent: #116466;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      gap: 18px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }}
    main {{ padding: 18px 24px 28px; }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
    }}
    table {{
      border-collapse: collapse;
      min-width: 1280px;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f6;
      z-index: 1;
      font-size: 12px;
      color: #344054;
    }}
    tr:hover {{ background: #f2f6f7; }}
    a {{ color: var(--accent); text-decoration: none; }}
    code {{
      display: block;
      max-width: 520px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 12px;
      color: #344054;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CryptoEA Optimization Dashboard</h1>
    <div class="meta">
      <span>{len(rows)} research runs</span>
      <span>Fees, stress execution, expectancy, and drawdown are read from saved JSON artifacts.</span>
    </div>
  </header>
  <main>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Symbol</th>
            <th>TF</th>
            <th>Family</th>
            <th>Trades</th>
            <th>PF</th>
            <th>Exp R</th>
            <th>DD R</th>
            <th>WF +Exp</th>
            <th>Score</th>
            <th>Params</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise SystemExit("No optimization summaries found.")
    write_csv(OPT_ROOT / "optimization_index.csv", rows)
    (OPT_ROOT / "dashboard.html").write_text(build_html(rows))
    print(f"Wrote {len(rows)} optimization rows to {OPT_ROOT / 'dashboard.html'}")


if __name__ == "__main__":
    main()
