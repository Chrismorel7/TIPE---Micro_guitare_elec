"""Compare time-domain magnetic pickup measurements with Faraday models."""

import logging
import sys
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Python"))
from library import model

DATA_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "raw" / "03-Reponse_mag"
OUTPUT_DIRECTORY = PROJECT_ROOT / "Python" / "Exports" / "03-Reponse_magnetique"

MU_0 = 4e-7 * np.pi
RELATIVE_PERMEABILITY = 12
SHAPE_FACTOR = 0.7
COIL_AREA_M2 = 1.5e-4
FIELD_FREQUENCY_HZ = 50
FIELD_AMPLITUDE_T = 3.68e-3
FIELD_GRADIENT_PER_M = 4
COIL_SPACING_M = 0.018
COIL_HEIGHT_M = 1.5e-2
SERIES_INDUCTANCE_H = 4.71
SINGLE_INDUCTANCE_H = 2.43
PICKUP_PHASES_RAD = {"single": 1.57, "series": 0.0}

SINGLE_MODEL_STYLE = {
    "color": "#FF0000",
    "label": "Single coil (model)",
    "linewidth": 3,
}
SERIES_MODEL_STYLE = {
    "color": "#009900",
    "label": "Humbucker series (model)",
    "linewidth": 3,
}
SINGLE_MEASUREMENT_STYLE = {
    "color": "#FF6666",
    "label": "Single coil (measurement)",
    "marker": "o",
    "markersize": 2,
    "linestyle": "none",
}
SERIES_MEASUREMENT_STYLE = {
    "color": "#66CC66",
    "label": "Humbucker series (measurement)",
    "marker": "o",
    "markersize": 2,
    "linestyle": "none",
}


def load_time_series(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate oscilloscope time and voltage samples."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=2, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read oscilloscope data from {filepath}") from error
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two time/voltage samples in {filepath}")

    times, voltage = data[:, 0], data[:, 1]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(voltage)):
        raise ValueError(f"Oscilloscope data contains non-finite values in {filepath}")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Sample times must be strictly increasing in {filepath}")
    return times, voltage


def calculate_turns(inductance_h: float) -> float:
    """Estimate coil turns from its inductance and physical dimensions."""
    if inductance_h <= 0:
        raise ValueError("Inductance must be positive")
    return float(
        np.sqrt(
            inductance_h
            * COIL_HEIGHT_M
            / (MU_0 * RELATIVE_PERMEABILITY * COIL_AREA_M2 * SHAPE_FACTOR)
        )
    )


def model_voltage(
    times: np.ndarray,
    frequency_hz: float,
    phase_rad: float,
    turns: float,
    gradient: float,
) -> np.ndarray:
    """Calculate induced voltage for a sinusoidal external magnetic field."""
    angular_frequency = 2 * np.pi * frequency_hz
    voltage_scale = (
        FIELD_AMPLITUDE_T
        * turns
        * COIL_AREA_M2
        * RELATIVE_PERMEABILITY
        * angular_frequency
        * gradient
    )
    return voltage_scale * np.sin(angular_frequency * times + phase_rad)


def main() -> None:
    """Plot measured and modeled induced voltages for both pickup types."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    single_times, single_voltage = load_time_series(DATA_DIRECTORY / "temp_sc.csv")
    series_times, series_voltage = load_time_series(DATA_DIRECTORY / "temp_hs.csv")
    single_turns = calculate_turns(SINGLE_INDUCTANCE_H)
    series_turns = calculate_turns(SERIES_INDUCTANCE_H) / 2

    single_model = model_voltage(
        single_times,
        FIELD_FREQUENCY_HZ,
        PICKUP_PHASES_RAD["single"],
        single_turns,
        1,
    )
    series_model = model_voltage(
        series_times,
        FIELD_FREQUENCY_HZ,
        PICKUP_PHASES_RAD["series"],
        series_turns,
        FIELD_GRADIENT_PER_M * COIL_SPACING_M,
    )
    model.save_plot(
        str(OUTPUT_DIRECTORY / "Singlecoil.png"),
        (single_times, single_model, SINGLE_MODEL_STYLE),
        (single_times, single_voltage, SINGLE_MEASUREMENT_STYLE),
        title="Pickup response to an external magnetic field",
        xlabel="Time [s]",
        ylabel="Induced voltage [V]",
        ylim=(-14, 14),
    )
    model.save_plot(
        str(OUTPUT_DIRECTORY / "Humbucker.png"),
        (series_times, series_model, SERIES_MODEL_STYLE),
        (series_times, series_voltage, SERIES_MEASUREMENT_STYLE),
        title="Pickup response to an external magnetic field",
        xlabel="Time [s]",
        ylabel="Induced voltage [V]",
        ylim=(-14, 14),
    )
    LOGGER.info("Magnetic-response figures saved to %s", OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
