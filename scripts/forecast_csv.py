"""Forecast future values from a saved run and history CSV.

Created by School of AI and School of QC.
"""

import argparse
from pathlib import Path

from quantum_reservoir_system.inference import forecast_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run saved recursive forecasts")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("future_forecasts.csv"))
    arguments = parser.parse_args()
    forecasts = forecast_csv(
        arguments.run_directory,
        arguments.history_csv,
        arguments.horizon,
    )
    forecasts.to_csv(arguments.output, index=False)
    print(f"Wrote {len(forecasts)} forecast steps to {arguments.output}")
    print("Created by School of AI and School of QC")


if __name__ == "__main__":
    main()
