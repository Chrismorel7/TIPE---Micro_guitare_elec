"""Calculate peak-to-peak amplitudes for filtered pickup measurements."""

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "filter"
OUTPUT_PATH = PROJECT_ROOT / "Python" / "data" / "filter" / "pickup_amplitudes.csv"
INPUT_FILES = tuple(
    f"{pickup}_par_{voltage}V.csv"
    for pickup in ("SingleCoil", "Humbucker")
    for voltage in ("2", "4", "6", "8", "10", "12,5", "15", "17,5", "20")
)


def get_amplitude(filepath: Path) -> float:
    """Return the peak-to-peak amplitude from a tab-delimited signal file."""
    try:
        signal_data = np.loadtxt(
            filepath,
            delimiter="\t",
            skiprows=1,
            usecols=1,
            max_rows=69_999,
            ndmin=1,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read signal data from {filepath}") from error

    if signal_data.size == 0 or not np.all(np.isfinite(signal_data)):
        raise ValueError(f"Signal data is empty or non-finite in {filepath}")
    return float(np.ptp(signal_data))


def write_amplitudes(output_path: Path, data_directory: Path) -> None:
    """Calculate all configured amplitudes and write a labeled CSV report."""
    rows = []
    for filename in INPUT_FILES:
        filepath = data_directory / filename
        label = filepath.stem.replace("_par_", " ")
        rows.append((label, get_amplitude(filepath)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Measurement", "Peak-to-peak amplitude"))
        writer.writerows(rows)


def main() -> None:
    """Generate the pickup amplitude report."""
    write_amplitudes(OUTPUT_PATH, DATA_DIRECTORY)


if __name__ == "__main__":
    main()
