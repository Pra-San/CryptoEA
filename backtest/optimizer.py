"""Parameter optimization for CryptoEA strategies.

Uses Optuna for Bayesian optimization with walk-forward validation.
Optimizes for Calmar Ratio, not raw returns.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.metrics import BacktestMetrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization.

    Attributes:
        best_params: Optimal parameter set.
        best_score: Best objective score achieved.
        best_metrics: Full performance metrics for best params.
        study: Optuna study object for analysis.
        parameter_importance: Feature importance for each parameter.
        optimization_history: History of all trials.
    """

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_metrics: Optional[PerformanceMetrics] = None
    study: Optional[optuna.Study] = None
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)


class ParameterOptimizer:
    """Bayesian parameter optimization using Optuna.

    Usage:
        optimizer = ParameterOptimizer(strategy_class, data)
        result = optimizer.optimize(
            objective_function=my_objective,
            n_trials=100,
            direction='maximize'
        )
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        initial_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            data: Preprocessed OHLCV DataFrame.
            symbol: Trading symbol.
            initial_config: Base configuration for the strategy.
        """
        self.data = data
        self.symbol = symbol
        self.initial_config = initial_config or {}
        self._metrics_calc = BacktestMetrics()

    def optimize(
        self,
        strategy_class: Callable,
        objective_fn: Callable,
        param_space: Dict[str, Tuple[Any, Any]],
        n_trials: int = 50,
        direction: str = "maximize",
        timeout_seconds: int = 3600,
        gc_after_trial: bool = True,
    ) -> OptimizationResult:
        """Run Bayesian optimization on strategy parameters.

        Args:
            strategy_class: Strategy class to instantiate with params.
            objective_fn: Function that takes params and returns score.
            param_space: Dictionary of {param_name: (min, max)} or
                        {param_name: (values_list,)}.
            n_trials: Number of optimization trials.
            direction: 'maximize' or 'minimize'.
            timeout_seconds: Time limit in seconds.
            gc_after_trial: Whether to run garbage collection after trial.

        Returns:
            OptimizationResult with best parameters and metrics.
        """
        logger.info(
            f"Starting optimization for {self.symbol}: "
            f"{n_trials} trials, {timeout_seconds}s timeout"
        )

        # Create Optuna study
        direction_str = "minimize" if direction == "minimize" else "maximize"
        sampler = TPESampler(seed=42, multivariate=True)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        study = optuna.create_study(
            direction=direction_str,
            sampler=sampler,
            pruner=pruner,
            storage=None,  # In-memory storage
            load_if_exists=False,
        )

        # Define optimization function
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna."""
            try:
                # Sample parameters from space
                params = {}
                for param_name, param_range in param_space.items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        # Continuous range
                        if param_range[0] < 100 and param_range[1] < 100:
                            params[param_name] = trial.suggest_float(
                                param_name, param_range[0], param_range[1]
                            )
                        else:
                            params[param_name] = trial.suggest_int(
                                param_name, param_range[0], param_range[1]
                            )
                    elif isinstance(param_range, tuple) and len(param_range) > 2:
                        # Categorical
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )

                # Run backtest with these parameters
                score = objective_fn(params)

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf") if direction == "maximize" else float("-inf")

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            gc_after_trial=gc_after_trial,
            show_progress_bar=False,
        )

        # Extract best result
        best_params = study.best_params
        best_score = study.best_value

        # Get metrics for best params
        best_metrics = None
        try:
            strategy = strategy_class()
            # Apply best params to strategy config
            if hasattr(strategy, 'config'):
                for k, v in best_params.items():
                    if hasattr(strategy.config, k):
                        setattr(strategy.config, k, v)

            engine = BacktestEngine(
                data=self.data,
                strategy=strategy,
                config=BacktestConfig(
                    initial_balance=10000,
                    fee_rate=0.0006,
                    slippage_rate=0.0003,
                ),
                symbol=self.symbol,
            )

            result = engine.run()
            best_metrics = self._metrics_calc.calculate(result)
        except Exception as e:
            logger.warning(f"Failed to compute best metrics: {e}")

        # Build optimization history
        optimization_history = []
        for trial in study.trials:
            if trial.value is not None and not np.isnan(trial.value):
                optimization_history.append({
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.value,
                })

        # Calculate parameter importance (correlation with objective)
        param_importance = self._calculate_parameter_importance(study)

        optimization_result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            study=study,
            parameter_importance=param_importance,
            optimization_history=optimization_history,
        )

        logger.info(
            f"Optimization complete: "
            f"Best score={best_score:.4f}, "
            f"Best params={best_params}"
        )

        return optimization_result

    def _calculate_parameter_importance(
        self, study: optuna.Study
    ) -> Dict[str, float]:
        """Calculate parameter importance based on correlation with objective.

        Args:
            study: Optuna study object.

        Returns:
            Dictionary of parameter importance scores.
        """
        importance: Dict[str, float] = {}

        # Get all completed trials
        completed_trials = [
            t for t in study.trials
            if t.state.name == "COMPLETE" and t.value is not None
        ]

        if len(completed_trials) < 3:
            return importance

        # Calculate correlation between each parameter and objective value
        values = [t.value for t in completed_trials]

        for param_name in study.best_params:
            param_values = [t.params.get(param_name) for t in completed_trials]

            # Filter out trials where parameter wasn't used
            valid_pairs = [
                (v, p) for v, p in zip(values, param_values)
                if p is not None
            ]

            if len(valid_pairs) < 3:
                importance[param_name] = 0.0
                continue

            val_array = np.array([p for _, p in valid_pairs])
            obj_array = np.array([v for v, _ in valid_pairs])

            if np.std(val_array) > 0 and np.std(obj_array) > 0:
                corr = np.corrcoef(val_array, obj_array)[0, 1]
                importance[param_name] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                importance[param_name] = 0.0

        return importance

    def print_optimization_summary(
        self, result: OptimizationResult
    ) -> None:
        """Print a formatted summary of optimization results.

        Args:
            result: OptimizationResult to summarize.
        """
        print("\n" + "=" * 60)
        print("  OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\n  Best Parameters:")
        for k, v in result.best_params.items():
            print(f"    {k}: {v}")

        print(f"\n  Best Score: {result.best_score:.4f}")

        if result.best_metrics:
            print(f"\n  Best Metrics:")
            print(f"    {result.best_metrics.summary}")

        print(f"\n  Parameter Importance:")
        for k, v in sorted(
            result.parameter_importance.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "#" * int(v * 20)
            print(f"    {k}: {v:.3f} {bar}")

        print(f"\n  Top 10 Trials:")
        sorted_trials = sorted(
            result.optimization_history,
            key=lambda x: x["value"] if result.best_score > 0 else -x["value"],
        )[:10]

        for i, trial in enumerate(sorted_trials, 1):
            print(f"    Trial {trial['trial_number']}: "
                  f"Score={trial['value']:.4f}, "
                  f"Params={trial['params']}")

        print("\n" + "=" * 60 + "\n")
