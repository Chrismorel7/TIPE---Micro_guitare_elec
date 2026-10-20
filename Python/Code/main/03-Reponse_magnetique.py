"""Compare pickup sensitivity to a spatially non-uniform magnetic field."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "raw" / "03-Reponse_mag"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "Python"
    / "Exports"
    / "03-Reponse_magnetique"
    / "Droite_de_sensibilite.png"
)

MU_0 = 4e-7 * np.pi
RELATIVE_PERMEABILITY = 2
SHAPE_FACTOR = 0.7
COIL_AREA_M2 = 5e-4
FIELD_FREQUENCY_HZ = 50
FIELD_GRADIENT_PER_M = 1
COIL_SPACING_M = 0.018
COIL_HEIGHT_M = 1.5e-2
HELMHOLTZ_TURNS = 95
HELMHOLTZ_RADIUS_M = 0.065
HELMHOLTZ_RADIUS_UNCERTAINTY_M = 0.005
INDUCTANCES_H = {"single": 2.43, "parallel": 1.77, "series": 4.71}


def load_measurement(filepath: Path) -> tuple[np.ndarray, ...]:
    """Load magnetic-field and induced-voltage measurements with uncertainties."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Unable to load magnetic-response data from {filepath}"
        ) from error

    if data.shape[0] < 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected an uncertainty row and measurement rows in {filepath}"
        )
    if not np.all(np.isfinite(data[:, :3])):
        raise ValueError(f"Measurement data contains non-finite values in {filepath}")

    current_uncertainty = data[0, 0] * np.sqrt(2)
    voltage_uncertainty = data[0, 2]
    currents = data[1:, 0] * np.sqrt(2)
    voltages = data[1:, 2]

    field_factor = (4 / 5) ** 1.5 * MU_0 * HELMHOLTZ_TURNS / HELMHOLTZ_RADIUS_M
    magnetic_field = field_factor * currents
    field_uncertainty = np.sqrt(
        (field_factor * current_uncertainty) ** 2
        + (magnetic_field * HELMHOLTZ_RADIUS_UNCERTAINTY_M / HELMHOLTZ_RADIUS_M) ** 2
    )
    return magnetic_field, voltages, field_uncertainty, voltage_uncertainty


def calculate_sensitivity(magnetic_field: np.ndarray, voltage: np.ndarray) -> float:
    """Calculate the least-squares sensitivity for a zero-intercept model."""
    if magnetic_field.size == 0 or magnetic_field.shape != voltage.shape:
        raise ValueError(
            "Magnetic-field and voltage arrays must be non-empty and aligned"
        )
    denominator = float(np.sum(magnetic_field**2))
    if denominator == 0:
        raise ValueError("Magnetic-field measurements must have non-zero magnitude")
    return float(np.sum(magnetic_field * voltage) / denominator)


def calculate_turns(inductance_h: float, winding_factor: float = 1.0) -> float:
    """Estimate winding turns from inductance and coil geometry."""
    if winding_factor <= 0:
        raise ValueError("winding_factor must be positive")
    return float(
        np.sqrt(
            winding_factor
            * inductance_h
            * COIL_HEIGHT_M
            / (MU_0 * RELATIVE_PERMEABILITY * COIL_AREA_M2 * SHAPE_FACTOR)
        )
    )


def plot_sensitivities(
    measurements: dict[str, tuple[np.ndarray, ...]],
    theoretical_slopes: dict[str, float],
) -> None:
    """Save experimental sensitivities and theoretical model lines."""
    field_max = max(float(np.max(values[0])) for values in measurements.values())
    field_axis = np.linspace(0, field_max * 1.1, 100)
    styles = {
        "single": ("#FF0000", "Single coil"),
        "parallel": ("#0000FF", "Humbucker parallel"),
        "series": ("#009900", "Humbucker series"),
    }

    figure, axis = plt.subplots(figsize=(6, 5))
    for name, (color, label) in styles.items():
        axis.plot(
            field_axis * 1e3,
            theoretical_slopes[name] * field_axis,
            color=color,
            linewidth=3,
            label=f"{label} (model)",
        )
        magnetic_field, voltage, field_uncertainty, voltage_uncertainty = measurements[
            name
        ]
        axis.errorbar(
            magnetic_field * 1e3,
            voltage,
            xerr=field_uncertainty * 1e3,
            yerr=voltage_uncertainty,
            fmt="none",
            color=color,
            capsize=5,
            label=f"{label} (measurement)",
        )

    axis.set_title("Pickup sensitivity to a 50 Hz magnetic field")
    axis.set_xlabel("Magnetic-field amplitude [mT]")
    axis.set_ylabel("Induced-voltage amplitude [V]")
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend(loc="upper left")
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300)
    plt.close(figure)


def main() -> None:
    """Run magnetic-response analysis and report sensitivity and CMRR results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    filenames = {
        "single": "13-05_S.csv",
        "parallel": "13-05_HP.csv",
        "series": "13-05_HS.csv",
    }
    measurements = {
        name: load_measurement(DATA_DIRECTORY / filename)
        for name, filename in filenames.items()
    }
    measured_slopes = {
        name: calculate_sensitivity(values[0], values[1])
        for name, values in measurements.items()
    }

    turns = {
        "single": calculate_turns(INDUCTANCES_H["single"]),
        "parallel": calculate_turns(INDUCTANCES_H["parallel"], winding_factor=2),
        "series": calculate_turns(INDUCTANCES_H["series"], winding_factor=0.5),
    }
    theoretical_slopes = {
        "single": (
            turns["single"]
            * COIL_AREA_M2
            * 2
            * np.pi
            * FIELD_FREQUENCY_HZ
            * RELATIVE_PERMEABILITY
        ),
        "parallel": (
            0.5
            * turns["parallel"]
            * COIL_AREA_M2
            * 2
            * np.pi
            * FIELD_FREQUENCY_HZ
            * RELATIVE_PERMEABILITY
            * FIELD_GRADIENT_PER_M
            * COIL_SPACING_M
        ),
        "series": (
            turns["series"]
            * COIL_AREA_M2
            * 2
            * np.pi
            * FIELD_FREQUENCY_HZ
            * RELATIVE_PERMEABILITY
            * FIELD_GRADIENT_PER_M
            * COIL_SPACING_M
        ),
    }

    plot_sensitivities(measurements, theoretical_slopes)
    for name in filenames:
        LOGGER.info(
            "%s sensitivity: model %.4g V/mT, measured %.4f V/mT",
            name,
            theoretical_slopes[name] * 1e-3,
            measured_slopes[name] * 1e-3,
        )

    single_sensitivity = measured_slopes["single"]
    for name in ("parallel", "series"):
        measured_ratio = abs(single_sensitivity / measured_slopes[name])
        theoretical_ratio = abs(theoretical_slopes["single"] / theoretical_slopes[name])
        LOGGER.info(
            "%s CMRR: measured %.1f dB, model %.1f dB",
            name,
            20 * np.log10(measured_ratio),
            20 * np.log10(theoretical_ratio),
        )


if __name__ == "__main__":
    main()
