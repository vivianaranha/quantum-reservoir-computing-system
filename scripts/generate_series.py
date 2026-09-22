"""Generate the deterministic demonstration series.

Created by School of AI and School of QC.
"""

from pathlib import Path

from quantum_reservoir_system.data import generate_mackey_glass


def main() -> None:
    destination = Path("data/mackey_glass.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_mackey_glass()
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows to {destination}")
    print("Created by School of AI and School of QC")


if __name__ == "__main__":
    main()
