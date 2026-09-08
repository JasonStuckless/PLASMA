"""Regenerate PLASMA plots from the current aggregate CSV outputs."""

from pathlib import Path

import pandas as pd

from plasma.plotting import save_metric_curves, save_pvp_barplots


CSV_DIR = Path("outputs/csv")
PLOT_DIR = Path("outputs/plots")


def main() -> None:
    aggregate_path = CSV_DIR / "aggregate_metrics.csv"
    pvp_path = CSV_DIR / "pvp_by_class.csv"

    if not aggregate_path.exists() or not pvp_path.exists():
        raise FileNotFoundError(
            "Expected outputs/csv/aggregate_metrics.csv and "
            "outputs/csv/pvp_by_class.csv. Run run_experiment.py first."
        )

    aggregate_df = pd.read_csv(aggregate_path)
    pvp_df = pd.read_csv(pvp_path)

    metric_paths = save_metric_curves(aggregate_df, PLOT_DIR)
    pvp_paths = save_pvp_barplots(pvp_df, PLOT_DIR)
    print(f"Regenerated {len(metric_paths) + len(pvp_paths)} plots in {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
