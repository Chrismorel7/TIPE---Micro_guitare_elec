"""Analyze FEMM magnetic-flux results and estimate induced voltage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "Docs" / "FEMM" / "resultats_flux.txt"
OUTPUT_PATH = (
    PROJECT_ROOT / "Python" / "Exports" / "03-Reponse_magnetique" / "femm_flux.png"
)
FREQUENCY_HZ = 78
COIL_TURNS = 1
SMOOTHING_METHOD = "SAVGOL"
COIL_HEIGHT_M = 0.015


def load_flux_data(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load angular position, displacement, and magnetic flux from FEMM output."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=2, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to load FEMM flux data from {filepath}") from error

    if data.shape[0] < 3 or data.shape[1] < 3:
        raise ValueError(f"Expected at least three rows and columns in {filepath}")
    if not np.all(np.isfinite(data[:, :3])):
        raise ValueError(f"FEMM flux data contains non-finite values in {filepath}")

    angles, displacement, flux = data[:, 0], data[:, 1], data[:, 2]
    angle_steps = np.diff(angles)
    if np.all(angle_steps < 0):
        return angles[::-1], displacement[::-1], flux[::-1]
    if np.any(angle_steps <= 0):
        raise ValueError("FEMM angular samples must be strictly monotonic")
    return angles, displacement, flux


def smooth_flux(
    times: np.ndarray, flux: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth flux samples and return their time derivative."""
    if method == "SAVGOL":
        window_length = min(51, flux.size if flux.size % 2 else flux.size - 1)
        if window_length < 5:
            raise ValueError(
                "At least five samples are required for Savitzky-Golay smoothing"
            )
        smoothed_flux = savgol_filter(flux, window_length=window_length, polyorder=3)
        flux_derivative = np.gradient(smoothed_flux, times)
    elif method == "SPLINE":
        spline = UnivariateSpline(times, flux, s=1e-15)
        smoothed_flux = spline(times)
        flux_derivative = spline.derivative()(times)
    elif method == "NONE":
        smoothed_flux = flux.copy()
        flux_derivative = np.gradient(flux, times)
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")
    return smoothed_flux, flux_derivative


def create_plot(
    times: np.ndarray,
    displacement: np.ndarray,
    flux: np.ndarray,
    smoothed_flux: np.ndarray,
    induced_voltage: np.ndarray,
    output_path: Path,
) -> None:
    """Plot displacement, flux, and induced voltage and save the figure."""
    figure, (position_axis, flux_axis, voltage_axis) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 10)
    )
    time_ms = times * 1_000

    position_axis.plot(time_ms, displacement, "k--", alpha=0.5, label="Displacement")
    position_axis.set_ylabel("Position [mm]")
    position_axis.set_title(f"FEMM simulation (f = {FREQUENCY_HZ} Hz)")
    position_axis.grid(True, linestyle="--", alpha=0.7)
    position_axis.legend(loc="upper right")

    flux_axis.plot(time_ms, flux * 1e6, "b.", markersize=2, alpha=0.3, label="Raw flux")
    flux_axis.plot(
        time_ms, smoothed_flux * 1e6, "g-", linewidth=2, label="Smoothed flux"
    )
    flux_axis.set_ylabel("Flux [µWb]")
    flux_axis.grid(True, linestyle="--", alpha=0.7)
    flux_axis.legend(loc="upper right")

    voltage_axis.plot(time_ms, induced_voltage * 1e3, "r-", linewidth=2)
    voltage_axis.set_ylabel("Voltage [mV]")
    voltage_axis.set_xlabel("Time [ms]")
    voltage_axis.grid(True, linestyle="--", alpha=0.7)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """Generate the FEMM displacement, flux, and induced-voltage plot."""
    angles, displacement, flux = load_flux_data(INPUT_PATH)
    angular_frequency = 2 * np.pi * FREQUENCY_HZ
    times = angles / angular_frequency
    smoothed_flux, flux_derivative = smooth_flux(times, flux, SMOOTHING_METHOD)
    induced_voltage = -COIL_TURNS * flux_derivative
    create_plot(
        times,
        displacement,
        flux,
        smoothed_flux,
        induced_voltage,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
