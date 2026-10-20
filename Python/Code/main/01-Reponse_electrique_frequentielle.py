"""Fit pickup impedance measurements and plot frequency responses."""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (
    PROJECT_ROOT
    / "Python"
    / "data"
    / "raw"
    / "01-Reponse_elec"
    / "Mesure_Impedance_PDT_100k.csv"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "Python" / "Exports" / "01-Reponse_electrique_frequetielle"
)
sys.path.insert(0, str(PROJECT_ROOT / "Python"))
from library import model

LOGGER = logging.getLogger(__name__)

SERIES_RESISTANCES = (5.26e3, 4.52e3, 15.34e3)
AMPLIFIER_RESISTANCE = 1e6
DIVIDER_RESISTANCE = 97.6e3
INITIAL_PARAMETERS = np.array([2.5, 0.5, 450.0])
LEARNING_RATE = 0.001
INPUT_VOLTAGE = 21.1
INPUT_VOLTAGE_UNCERTAINTY = 0.01
MEASURED_VOLTAGE_UNCERTAINTY = 0.01
DIVIDER_RESISTANCE_UNCERTAINTY = DIVIDER_RESISTANCE * 0.01
PICKUP_NAMES = ("Single coil", "Humbucker parallel", "Humbucker series")
THEORY_STYLES = (
    {
        "color": "#FF0000",
        "label": "Single coil (fit)",
        "linewidth": 3,
        "linestyle": ":",
    },
    {
        "color": "#0000FF",
        "label": "Humbucker parallel (fit)",
        "linewidth": 3,
        "linestyle": ":",
    },
    {
        "color": "#009900",
        "label": "Humbucker series (fit)",
        "linewidth": 3,
        "linestyle": ":",
    },
)
EXPERIMENT_STYLES = (
    {
        "color": "#FF6666",
        "label": "Single coil (measurement)",
        "marker": "o",
        "markersize": 6,
        "linestyle": "none",
    },
    {
        "color": "#6666FF",
        "label": "Humbucker parallel (measurement)",
        "marker": "o",
        "markersize": 6,
        "linestyle": "none",
    },
    {
        "color": "#66CC66",
        "label": "Humbucker series (measurement)",
        "marker": "o",
        "markersize": 6,
        "linestyle": "none",
    },
)


def load_measurements(filepath: Path) -> tuple[np.ndarray, ...]:
    """Load frequency and measured pickup voltages from the experiment CSV."""
    try:
        measurements = np.loadtxt(filepath, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Unable to read impedance measurements from {filepath}"
        ) from error

    if measurements.shape[0] == 0 or measurements.shape[1] < 5:
        raise ValueError(f"Expected frequency and four voltage columns in {filepath}")
    if not np.all(np.isfinite(measurements[:, :5])):
        raise ValueError(f"Measurement data contains non-finite values in {filepath}")

    frequencies = measurements[:, 0]
    return frequencies, measurements[:, 2], measurements[:, 3], measurements[:, 4]


def calculate_impedance_uncertainty(
    divider_resistance: float,
    divider_uncertainty: float,
    input_voltage: float,
    input_voltage_uncertainty: float,
    measured_voltage: np.ndarray,
    measured_voltage_uncertainty: float,
) -> np.ndarray:
    """Propagate voltage and resistor uncertainties to the impedance estimate."""
    if divider_resistance <= 0 or input_voltage <= 0 or np.any(measured_voltage <= 0):
        raise ValueError("Resistance and measured voltages must be positive")

    relative_uncertainty = np.sqrt(
        (divider_uncertainty / divider_resistance) ** 2
        + (input_voltage_uncertainty / input_voltage) ** 2
        + (measured_voltage_uncertainty / measured_voltage) ** 2
    )
    impedance = divider_resistance * (input_voltage / measured_voltage - 1)
    return np.abs(impedance) * relative_uncertainty


def calculate_impedance(measured_voltage: np.ndarray) -> np.ndarray:
    """Estimate pickup impedance from the voltage-divider measurements."""
    return DIVIDER_RESISTANCE * (INPUT_VOLTAGE / measured_voltage - 1)


def fit_pickups(
    frequencies: np.ndarray, measured_impedances: tuple[np.ndarray, ...]
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fit the three pickup models and return parameters and normalized costs."""
    iteration_count = 50
    costs = np.zeros((len(SERIES_RESISTANCES), iteration_count))
    fitted_parameters = []

    for pickup_index, (series_resistance, impedance) in enumerate(
        zip(SERIES_RESISTANCES, measured_impedances)
    ):
        parameters = INITIAL_PARAMETERS.copy()
        for iteration in range(iteration_count):
            parameters, costs[pickup_index, iteration] = model.moindres_carres(
                INITIAL_PARAMETERS,
                series_resistance,
                frequencies,
                impedance,
                iteration + 1,
                LEARNING_RATE,
            )
        fitted_parameters.append(parameters)
        costs[pickup_index] = model.normalize_signal(costs[pickup_index])

    return fitted_parameters, costs


def save_bode_plot(
    frequencies: np.ndarray, responses: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    """Save the fitted pickup gain and phase as a Bode diagram."""
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for axis in axes:
        axis.axvspan(82.41, 1174.66, color="#ADD8E6", alpha=0.6, zorder=0)
        axis.axvspan(
            1174.66,
            5 * 1174.66,
            facecolor="none",
            alpha=0.5,
            hatch="////",
            edgecolor="#424242",
            zorder=0,
        )
        axis.set_xscale("log")
        axis.grid(which="both", linestyle="--", linewidth=0.5)
        axis.set_xlim(10, 30_000)

    for (gain, phase), style in zip(responses, THEORY_STYLES):
        axes[0].plot(frequencies, gain, **style)
        axes[1].plot(frequencies, phase, **style)

    axes[0].set_ylabel("Gain [dB]")
    axes[0].set_title("Pickup Bode diagram")
    axes[1].set_ylabel("Phase [°]")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "00-Bode.png", dpi=300)
    plt.close(figure)


def main() -> None:
    """Run the electrical-response analysis and save its figures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frequencies, *measured_voltages = load_measurements(DATA_PATH)
    measured_impedances = tuple(
        calculate_impedance(voltage) for voltage in measured_voltages
    )
    uncertainty = tuple(
        calculate_impedance_uncertainty(
            DIVIDER_RESISTANCE,
            DIVIDER_RESISTANCE_UNCERTAINTY,
            INPUT_VOLTAGE,
            INPUT_VOLTAGE_UNCERTAINTY,
            voltage,
            MEASURED_VOLTAGE_UNCERTAINTY,
        )
        * 1e-3
        for voltage in measured_voltages
    )

    parameters, convergence = fit_pickups(frequencies, measured_impedances)
    model_frequencies = np.linspace(10, 30_000, 1_000)
    responses = [
        model.calcul_bode(fit, resistance, model_frequencies, AMPLIFIER_RESISTANCE)
        for fit, resistance in zip(parameters, SERIES_RESISTANCES)
    ]

    model.save_plot(
        str(OUTPUT_DIRECTORY / "Convergence.png"),
        *(
            (range(1, convergence.shape[1] + 1), convergence[index] * 100, style)
            for index, style in enumerate(THEORY_STYLES)
        ),
        title="Least-squares convergence",
        xlabel="Iteration",
        ylabel="Normalized error [%]",
    )
    save_bode_plot(model_frequencies, responses)

    impedance_curves = []
    for index, (resistance, fit) in enumerate(zip(SERIES_RESISTANCES, parameters)):
        impedance_curves.extend(
            (
                (
                    model_frequencies,
                    model.pickup_impedance(
                        fit, resistance, 2 * np.pi * model_frequencies
                    )
                    * 1e-3,
                    THEORY_STYLES[index],
                ),
                (
                    frequencies,
                    measured_impedances[index] * 1e-3,
                    {**EXPERIMENT_STYLES[index], "yerr": uncertainty[index]},
                ),
            )
        )
        LOGGER.info(
            "%s fit: L=%.2f H, C=%.2f nF, Rp=%.1f kOhm",
            PICKUP_NAMES[index],
            *fit,
        )

    model.save_plot(
        str(OUTPUT_DIRECTORY / "Impedance.png"),
        *impedance_curves,
        title="Pickup impedance magnitude",
        xlabel="Frequency [Hz]",
        ylabel="Impedance [kOhm]",
        vspan=True,
        xlim=(10, 30_000),
        log=True,
        error_bars=True,
    )


if __name__ == "__main__":
    main()
