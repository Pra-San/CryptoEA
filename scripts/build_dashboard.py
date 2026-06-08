#!/usr/bin/env python3
"""Build a static dashboard from reproducible backtest run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "research" / "backtests"


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Load all summary.json files under the run artifact root."""
    runs: list[dict[str, Any]] = []
    for summary_path in sorted((root / "runs").glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        summary["_summary_path"] = str(summary_path.relative_to(root))
        runs.append(summary)
    return runs


def metric(run: dict[str, Any], name: str, default: Any = 0) -> Any:
    return run.get("metrics", {}).get(name, default)


def write_index_csv(root: Path, runs: list[dict[str, Any]]) -> None:
    """Write a flat CSV index for quick spreadsheet review."""
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "validation_mode",
        "symbol",
        "timeframe",
        "strategy_version",
        "start",
        "end",
        "total_trades",
        "win_rate",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "profit_factor",
        "max_drawdown_pct",
        "max_drawdown_r",
        "avg_drawdown_r",
        "trades_per_day",
        "avg_win",
        "avg_loss",
        "avg_win_r",
        "avg_loss_r",
        "avg_r_multiple",
        "expectancy_r",
        "payoff_ratio",
        "expectancy",
        "avg_holding_bars",
        "avg_trade_return",
        "total_return_pct",
        "cagr_pct",
        "max_consecutive_losses",
        "exposure_pct",
        "fee_rate",
        "slippage_rate",
    ]

    with open(root / "run_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            period = run.get("period", {})
            row = {
                "run_id": run.get("run_id"),
                "validation_mode": run.get("validation_mode"),
                "symbol": run.get("symbol"),
                "timeframe": run.get("timeframe"),
                "strategy_version": run.get("strategy_version"),
                "start": period.get("start"),
                "end": period.get("end"),
                "fee_rate": run.get("engine_config", {}).get("fee_rate"),
                "slippage_rate": run.get("engine_config", {}).get("slippage_rate"),
            }
            row.update({field: metric(run, field) for field in fields if field not in row})
            writer.writerow(row)


def build_html(runs: list[dict[str, Any]]) -> str:
    """Render the dashboard HTML."""
    payload = json.dumps(runs, separators=(",", ":"))
    generated_count = len(runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CryptoEA Backtest Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f3;
      --surface: #ffffff;
      --surface-2: #eef2ef;
      --line: #d7ddd8;
      --ink: #182026;
      --muted: #5e6b73;
      --teal: #0f766e;
      --blue: #2563eb;
      --amber: #b45309;
      --red: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .topline {{
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(460px, 1fr) minmax(360px, 520px);
      gap: 18px;
      padding: 18px;
    }}
    section {{
      min-width: 0;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    input, select {{
      height: 36px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      padding: 0 10px;
      border-radius: 6px;
      font-size: 14px;
    }}
    input {{ min-width: 220px; flex: 1; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 73px;
      background: var(--surface-2);
      color: var(--muted);
      font-weight: 700;
      z-index: 2;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:nth-child(3), td:nth-child(3) {{
      text-align: left;
    }}
    tr {{
      cursor: pointer;
    }}
    tr:hover td {{
      background: #f9faf7;
    }}
    tr.selected td {{
      background: #e8f3f1;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 16px;
      min-height: 260px;
    }}
    .panel h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 13px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0;
    }}
    .metric {{
      background: var(--surface-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 76px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 18px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }}
    .positive {{ color: var(--teal); }}
    .warning {{ color: var(--amber); }}
    .negative {{ color: var(--red); }}
    .bars {{
      display: grid;
      gap: 8px;
      margin: 14px 0;
    }}
    .barrow {{
      display: grid;
      grid-template-columns: 126px 1fr 72px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .track {{
      height: 10px;
      background: #e3e8e3;
      overflow: hidden;
      border-radius: 999px;
    }}
    .fill {{
      height: 100%;
      background: var(--blue);
      width: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 8px 12px;
      margin: 16px 0 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .empty {{
      background: var(--surface);
      border: 1px dashed var(--line);
      padding: 24px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .topline {{ white-space: normal; }}
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      th {{ top: 0; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 620px) {{
      table {{ font-size: 12px; }}
      th, td {{ padding: 8px 7px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .barrow {{ grid-template-columns: 104px 1fr 58px; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CryptoEA Backtest Dashboard</h1>
    <div class="topline"><span id="runCount">{generated_count} runs</span><span id="selectedLabel"></span></div>
  </header>
  <main>
    <section>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Search runs">
        <select id="mode"><option value="">All modes</option></select>
        <select id="symbol"><option value="">All symbols</option></select>
        <select id="version"><option value="">All versions</option></select>
      </div>
      <div id="tableMount"></div>
    </section>
    <section>
      <div id="detail" class="panel"></div>
    </section>
  </main>
  <script>
    const runs = {payload};
    const fmt = new Intl.NumberFormat(undefined, {{ maximumFractionDigits: 2 }});
    const pct = value => `${{fmt.format(Number(value || 0))}}%`;
    const num = value => fmt.format(Number(value || 0));
    const metric = (run, key) => run.metrics?.[key] ?? 0;
    const rateBps = value => `${{fmt.format(Number(value || 0) * 10000)}} bps`;
    let selectedId = runs[0]?.run_id || null;

    const filters = {{
      search: document.getElementById('search'),
      mode: document.getElementById('mode'),
      symbol: document.getElementById('symbol'),
      version: document.getElementById('version'),
    }};

    function fillOptions(id, values) {{
      const select = filters[id];
      values.forEach(value => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}

    fillOptions('mode', [...new Set(runs.map(r => r.validation_mode).filter(Boolean))].sort());
    fillOptions('symbol', [...new Set(runs.map(r => r.symbol).filter(Boolean))].sort());
    fillOptions('version', [...new Set(runs.map(r => r.strategy_version).filter(Boolean))].sort());

    Object.values(filters).forEach(el => el.addEventListener('input', render));

    function filteredRuns() {{
      const q = filters.search.value.toLowerCase().trim();
      return runs.filter(run => {{
        const haystack = [run.run_id, run.symbol, run.timeframe, run.strategy, run.strategy_version, run.validation_mode].join(' ').toLowerCase();
        return (!q || haystack.includes(q))
          && (!filters.mode.value || run.validation_mode === filters.mode.value)
          && (!filters.symbol.value || run.symbol === filters.symbol.value)
          && (!filters.version.value || run.strategy_version === filters.version.value);
      }});
    }}

    function render() {{
      const rows = filteredRuns();
      document.getElementById('runCount').textContent = `${{rows.length}} / ${{runs.length}} runs`;
      renderTable(rows);
      if (!rows.some(r => r.run_id === selectedId)) selectedId = rows[0]?.run_id || null;
      renderDetail(rows.find(r => r.run_id === selectedId) || rows[0]);
    }}

    function renderTable(rows) {{
      const mount = document.getElementById('tableMount');
      if (!rows.length) {{
        mount.innerHTML = '<div class="empty">No runs match the current filters.</div>';
        return;
      }}
      const sorted = [...rows].sort((a, b) => {{
        const mode = String(a.validation_mode).localeCompare(String(b.validation_mode));
        if (mode) return mode;
        return String(a.run_id).localeCompare(String(b.run_id));
      }});
      mount.innerHTML = `<table>
        <thead><tr>
          <th>Run</th><th>Mode</th><th>Market</th><th>Trades</th><th>WR</th><th>Sharpe</th><th>PF</th><th>Max DD</th><th>Max DD R</th><th>Trades/Day</th>
        </tr></thead>
        <tbody>${{sorted.map(run => `
          <tr data-id="${{escapeHtml(run.run_id)}}" class="${{run.run_id === selectedId ? 'selected' : ''}}">
            <td>${{escapeHtml(run.run_id)}}</td>
            <td>${{escapeHtml(run.validation_mode || '')}}</td>
            <td>${{escapeHtml(run.symbol)}} ${{escapeHtml(run.timeframe)}}</td>
            <td>${{num(metric(run, 'total_trades'))}}</td>
            <td>${{pct(metric(run, 'win_rate') * 100)}}</td>
            <td>${{num(metric(run, 'sharpe_ratio'))}}</td>
            <td>${{num(metric(run, 'profit_factor'))}}</td>
            <td class="${{metric(run, 'max_drawdown_pct') > 10 ? 'negative' : 'warning'}}">${{pct(metric(run, 'max_drawdown_pct'))}}</td>
            <td class="${{metric(run, 'max_drawdown_r') > 6 ? 'negative' : 'warning'}}">${{num(metric(run, 'max_drawdown_r'))}}R</td>
            <td>${{num(metric(run, 'trades_per_day'))}}</td>
          </tr>`).join('')}}</tbody>
      </table>`;
      mount.querySelectorAll('tr[data-id]').forEach(row => {{
        row.addEventListener('click', () => {{
          selectedId = row.dataset.id;
          render();
        }});
      }});
    }}

    function renderDetail(run) {{
      const detail = document.getElementById('detail');
      if (!run) {{
        detail.innerHTML = '<div class="empty">No run selected.</div>';
        document.getElementById('selectedLabel').textContent = '';
        return;
      }}
      document.getElementById('selectedLabel').textContent = run.run_id;
      const artifacts = run.artifacts || {{}};
      detail.innerHTML = `
        <h2>${{escapeHtml(run.symbol)}} ${{escapeHtml(run.timeframe)}} ${{escapeHtml(run.strategy_version)}}</h2>
        <div class="subtle">${{escapeHtml(run.validation_mode)}} · ${{escapeHtml(run.period?.start || '')}} → ${{escapeHtml(run.period?.end || '')}}</div>
        <div class="metrics">
          ${{metricTile('Total Return', pct(metric(run, 'total_return_pct')), metric(run, 'total_return_pct') >= 0 ? 'positive' : 'negative')}}
          ${{metricTile('Max Drawdown', pct(metric(run, 'max_drawdown_pct')), metric(run, 'max_drawdown_pct') > 10 ? 'negative' : 'warning')}}
          ${{metricTile('Max DD R', `${{num(metric(run, 'max_drawdown_r'))}}R`, metric(run, 'max_drawdown_r') > 6 ? 'negative' : 'warning')}}
          ${{metricTile('Profit Factor', num(metric(run, 'profit_factor')), metric(run, 'profit_factor') >= 1.5 ? 'positive' : 'negative')}}
          ${{metricTile('Sharpe', num(metric(run, 'sharpe_ratio')), metric(run, 'sharpe_ratio') >= 1.2 ? 'positive' : 'negative')}}
          ${{metricTile('Trades/Day', num(metric(run, 'trades_per_day')), '')}}
          ${{metricTile('Max Loss Streak', num(metric(run, 'max_consecutive_losses')), metric(run, 'max_consecutive_losses') > 5 ? 'negative' : '')}}
          ${{metricTile('Avg Win', num(metric(run, 'avg_win')), 'positive')}}
          ${{metricTile('Avg Loss', num(metric(run, 'avg_loss')), 'negative')}}
          ${{metricTile('Avg RR', num(metric(run, 'payoff_ratio')), metric(run, 'payoff_ratio') >= 1 ? 'positive' : 'negative')}}
          ${{metricTile('Avg R', `${{num(metric(run, 'avg_r_multiple'))}}R`, metric(run, 'avg_r_multiple') > 0 ? 'positive' : 'negative')}}
          ${{metricTile('Expectancy R', `${{num(metric(run, 'expectancy_r'))}}R`, metric(run, 'expectancy_r') > 0 ? 'positive' : 'negative')}}
        </div>
        <div class="bars">
          ${{bar('Win Rate', metric(run, 'win_rate') * 100, 100, pct(metric(run, 'win_rate') * 100), 'var(--teal)')}}
          ${{bar('Exposure', metric(run, 'exposure_pct') * 100, 100, pct(metric(run, 'exposure_pct') * 100), 'var(--blue)')}}
          ${{bar('OOS DD', metric(run, 'max_drawdown_pct'), 15, pct(metric(run, 'max_drawdown_pct')), 'var(--amber)')}}
          ${{bar('R DD', metric(run, 'max_drawdown_r'), 10, `${{num(metric(run, 'max_drawdown_r'))}}R`, 'var(--amber)')}}
        </div>
        <dl>
          <dt>Run ID</dt><dd>${{escapeHtml(run.run_id)}}</dd>
          <dt>Git SHA</dt><dd>${{escapeHtml(run.git_sha || '')}}</dd>
          <dt>Bars</dt><dd>${{num(run.data?.bars)}}</dd>
          <dt>Fee Rate</dt><dd>${{rateBps(run.engine_config?.fee_rate)}} per side</dd>
          <dt>Adverse Exec</dt><dd>${{rateBps(run.engine_config?.slippage_rate)}} per side</dd>
          <dt>Expectancy</dt><dd>${{num(metric(run, 'expectancy'))}}</dd>
          <dt>Expectancy R</dt><dd>${{num(metric(run, 'expectancy_r'))}}R</dd>
          <dt>Avg Win R</dt><dd>${{num(metric(run, 'avg_win_r'))}}R</dd>
          <dt>Avg Loss R</dt><dd>${{num(metric(run, 'avg_loss_r'))}}R</dd>
          <dt>Avg Holding</dt><dd>${{num(metric(run, 'avg_holding_bars'))}} bars</dd>
          <dt>Avg Trade Return</dt><dd>${{pct(metric(run, 'avg_trade_return') * 100)}}</dd>
          <dt>Total Fees</dt><dd>${{num(metric(run, 'total_fees'))}}</dd>
          <dt>Worst Trade</dt><dd>${{num(metric(run, 'worst_trade'))}}</dd>
          <dt>Best Month</dt><dd>${{pct(metric(run, 'best_month') * 100)}}</dd>
          <dt>Worst Month</dt><dd>${{pct(metric(run, 'worst_month') * 100)}}</dd>
          <dt>Report</dt><dd>${{artifacts.report_txt ? `<a href="${{escapeAttr(artifacts.report_txt)}}">report.txt</a>` : ''}}</dd>
          <dt>Trades</dt><dd>${{artifacts.trades_csv ? `<a href="${{escapeAttr(artifacts.trades_csv)}}">trades.csv</a>` : ''}}</dd>
          <dt>Equity</dt><dd>${{artifacts.equity_csv ? `<a href="${{escapeAttr(artifacts.equity_csv)}}">equity.csv</a>` : ''}}</dd>
        </dl>`;
    }}

    function metricTile(label, value, klass) {{
      return `<div class="metric"><div class="label">${{escapeHtml(label)}}</div><div class="value ${{klass}}">${{escapeHtml(value)}}</div></div>`;
    }}

    function bar(label, value, max, shown, color) {{
      const width = Math.max(0, Math.min(100, Number(value || 0) / max * 100));
      return `<div class="barrow"><div>${{escapeHtml(label)}}</div><div class="track"><div class="fill" style="width:${{width}}%;background:${{color}}"></div></div><div>${{escapeHtml(shown)}}</div></div>`;
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value).replace(/`/g, '&#96;');
    }}

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CryptoEA backtest dashboard")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    output = args.output or root / "dashboard.html"
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    runs = load_runs(root)
    write_index_csv(root, runs)
    output.write_text(build_html(runs))
    print(f"Dashboard written to {output}")


if __name__ == "__main__":
    main()
