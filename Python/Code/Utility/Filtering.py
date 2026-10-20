"""Filter experimental oscilloscope signals and save comparison plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "raw" / "02-Reponse_meca"
OUTPUT_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "filter" / "02-Reponse_meca"
INPUT_FILES = (
    "Hp1.csv",
    "Hp2.csv",
    "Hp3.csv",
    "Hs1.csv",
    "Hs2.csv",
    "Hs3.csv",
    "S1.csv",
    "S2.csv",
)


def read_signal(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the time and voltage columns from an oscilloscope CSV file."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=2, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Unable to read numeric signal data from {filepath}"
        ) from error

    if data.shape[1] < 2 or data.shape[0] == 0:
        raise ValueError(f"Expected at least one time/voltage sample in {filepath}")

    times, signal = data[:, 0], data[:, 1]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(signal)):
        raise ValueError(f"Signal data contains non-finite values in {filepath}")
    return times, signal


def write_signal(filepath: Path, times: np.ndarray, signal: np.ndarray) -> None:
    """Write aligned time and voltage samples as a tab-delimited text file."""
    if times.size != signal.size:
        raise ValueError("Time and signal arrays must have the same length")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8", newline="") as output_file:
        output_file.write("Time\tSignal\n")
        np.savetxt(output_file, np.column_stack((times, signal)), delimiter="\t")


def moving_average(signal: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Apply a centered moving average while preserving the input length."""
    if window_size < 1:
        raise ValueError("window_size must be a positive integer")
    window_size = min(int(window_size), signal.size)
    if signal.size == 0:
        raise ValueError("signal must contain at least one sample")

    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(signal, kernel, mode="same")


def first_order_low_pass(
    times: np.ndarray, signal: np.ndarray, time_constant: float = 0.001
) -> np.ndarray:
    """Apply a causal first-order low-pass filter to a uniformly sampled signal."""
    if times.size != signal.size or signal.size == 0:
        raise ValueError(
            "Time and signal arrays must be non-empty and have equal lengths"
        )
    if time_constant <= 0:
        raise ValueError("time_constant must be positive")
    if times.size > 1:
        intervals = np.diff(times)
        if np.any(intervals <= 0):
            raise ValueError("Sample times must be strictly increasing")
        dt = float(np.mean(intervals))
    else:
        dt = 1.0

    alpha = dt / (time_constant + dt)
    filtered_signal = np.empty_like(signal, dtype=float)
    filtered_signal[0] = signal[0]
    for index in range(1, signal.size):
        filtered_signal[index] = (
            alpha * signal[index] + (1 - alpha) * filtered_signal[index - 1]
        )
    return filtered_signal


def compare_filters(
    input_path: Path,
    output_path: Path,
    window_size: int = 100,
    time_constant: float = 0.001,
) -> None:
    """Plot raw and filtered signals, then save the low-pass result."""
    times, signal = read_signal(input_path)
    averaged_signal = moving_average(signal, window_size)
    low_pass_signal = first_order_low_pass(times, signal, time_constant)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(times, signal, label="Raw signal", color="#636363", linewidth=1)
    axis.plot(
        times, averaged_signal, label="Moving average", color="#EE2222", linewidth=2
    )
    axis.plot(
        times,
        low_pass_signal,
        label="First-order low-pass",
        color="#0066CC",
        linewidth=2,
    )
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Induced voltage [V]")
    axis.set_title("Filter comparison")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_name(f"{output_path.stem}_filter_plot.png"))
    plt.close(figure)
    write_signal(output_path, times, low_pass_signal)


def main() -> None:
    """Filter each configured mechanical-response measurement."""
    for filename in INPUT_FILES:
        input_path = INPUT_DIRECTORY / filename
        output_path = OUTPUT_DIRECTORY / filename
        compare_filters(input_path, output_path)


if __name__ == "__main__":
    main()
