"""Compare measured and modeled mechanical pickup responses."""

import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Python"))
from library import model

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = PROJECT_ROOT / "Python" / "data" / "raw" / "02-Reponse_meca"
OUTPUT_DIRECTORY = PROJECT_ROOT / "Python" / "Exports" / "02-Reponse_mecanique"
MODE_COUNT = 20
DURATION_S = 1.0
SAMPLE_COUNT = 10_000
MEASUREMENT_UNCERTAINTY_V = 1e-5 / 2
PEAK_PROMINENCE = 0.03

MU_0 = 4e-7 * np.pi
RELATIVE_PERMEABILITY = 3.5
SHAPE_FACTOR = 0.7
COIL_AREA_M2 = 1.5e-4
COIL_HEIGHT_M = 1.5e-2
PARALLEL_POSITIONS_M = (0.147, 0.165)
SERIES_POSITIONS_M = (0.202, 0.220)
INITIAL_DISPLACEMENT_M = 1e-3
DISTURBANCE_POSITION_M = 0.300
FUNDAMENTAL_FREQUENCY_HZ = 80
FILTER_FREQUENCY_RANGE_HZ = (40, 800)

INDUCTANCE_SINGLE_H = 2.43
INDUCTANCE_PARALLEL_H = 1.77
INDUCTANCE_SERIES_H = 4.71
TURNS_PARALLEL = np.sqrt(
    2
    * INDUCTANCE_PARALLEL_H
    * COIL_HEIGHT_M
    / (MU_0 * RELATIVE_PERMEABILITY * COIL_AREA_M2 * SHAPE_FACTOR)
)
TURNS_SERIES = np.sqrt(
    INDUCTANCE_SERIES_H
    * COIL_HEIGHT_M
    / (2 * MU_0 * RELATIVE_PERMEABILITY * COIL_AREA_M2 * SHAPE_FACTOR)
)

THEORY_STYLES = (
    {"color": "#0000FF", "label": "Humbucker parallel (model)", "linewidth": 3},
    {"color": "#FF0000", "label": "Single coil (model)", "linewidth": 3},
    {"color": "#009900", "label": "Humbucker series (model)", "linewidth": 3},
)
EXPERIMENT_STYLES = (
    {
        "color": "#6666FF",
        "label": "Humbucker parallel (measurement)",
        "marker": "o",
        "markersize": 8,
        "linestyle": ":",
    },
    {
        "color": "#FF6666",
        "label": "Single coil (measurement)",
        "marker": "o",
        "markersize": 8,
        "linestyle": ":",
    },
    {
        "color": "#66CC66",
        "label": "Humbucker series (measurement)",
        "marker": "o",
        "markersize": 8,
        "linestyle": ":",
    },
)
SPECTRUM_TITLE = "Induced pickup voltage spectrum"
ENVELOPE_STYLE = {"color": "k", "linestyle": "--", "label": "1/n envelope"}


def load_signal(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate time and voltage samples from an oscilloscope CSV."""
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


def get_frequency_axis(sample_times: np.ndarray) -> np.ndarray:
    """Calculate the FFT frequency bins for evenly sampled time data."""
    if sample_times.size < 2:
        raise ValueError("At least two time samples are required for the FFT")
    intervals = np.diff(sample_times)
    if not np.allclose(intervals, intervals[0], rtol=1e-3, atol=0):
        raise ValueError("FFT analysis requires uniformly sampled data")
    return np.fft.fftfreq(sample_times.size, d=float(np.mean(intervals)))


def plot_spectrum(
    output_path: Path,
    datasets: tuple[tuple[np.ndarray, np.ndarray, dict], ...],
    envelope: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Save one spectrum plot, optionally including the reference envelope."""
    plot_calls = list(datasets)
    if envelope is not None:
        plot_calls.append((envelope[0], envelope[1], ENVELOPE_STYLE))
    model.save_plot(
        str(output_path),
        *plot_calls,
        title=SPECTRUM_TITLE,
        xlabel="Frequency [Hz]",
        ylabel="Normalized voltage",
        error_bars=True,
    )


def analyze_spectrum(
    times: np.ndarray, voltage: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return positive-frequency spectral amplitudes and their full-spectrum peak."""
    spectrum = model.compute_and_normalize_fft(voltage, voltage.size)
    peak_amplitude = float(np.max(spectrum))
    if peak_amplitude <= 0:
        raise ValueError("Cannot normalize an empty or zero-amplitude spectrum")
    frequencies = get_frequency_axis(times)
    mask = (frequencies > FILTER_FREQUENCY_RANGE_HZ[0]) & (
        frequencies <= FILTER_FREQUENCY_RANGE_HZ[1]
    )
    return frequencies[mask], spectrum[mask], peak_amplitude


def create_spectrum_plots() -> None:
    """Generate model, measurement, and combined FFT figures."""
    experimental = {
        "single": load_signal(DATA_DIRECTORY / "S2.csv"),
        "parallel": load_signal(DATA_DIRECTORY / "Hp2.csv"),
        "series": load_signal(DATA_DIRECTORY / "Hs3.csv"),
    }
    theoretical_times = np.linspace(-DURATION_S / 2, DURATION_S / 2, SAMPLE_COUNT)
    theoretical_signals = {
        "single": model.fem_singlecoil(
            MODE_COUNT,
            theoretical_times,
            INITIAL_DISPLACEMENT_M,
            DISTURBANCE_POSITION_M,
        ),
        "parallel": model.fem_humbucker(
            MODE_COUNT,
            TURNS_PARALLEL,
            *PARALLEL_POSITIONS_M,
            theoretical_times,
            INITIAL_DISPLACEMENT_M,
            DISTURBANCE_POSITION_M,
        ),
        "series": model.fem_humbucker(
            MODE_COUNT,
            TURNS_SERIES,
            *SERIES_POSITIONS_M,
            theoretical_times,
            INITIAL_DISPLACEMENT_M,
            DISTURBANCE_POSITION_M,
        ),
    }

    theoretical_frequencies = get_frequency_axis(theoretical_times)
    theoretical_spectra = {
        name: model.compute_and_normalize_fft(signal, signal.size)
        for name, signal in theoretical_signals.items()
    }
    frequency_mask = (theoretical_frequencies > FILTER_FREQUENCY_RANGE_HZ[0]) & (
        theoretical_frequencies <= FILTER_FREQUENCY_RANGE_HZ[1]
    )
    model_frequencies = theoretical_frequencies[frequency_mask]
    model_spectra = {
        name: model.normalize_signal(spectrum[frequency_mask])
        for name, spectrum in theoretical_spectra.items()
    }

    experimental_spectra = {}
    for name, (times, voltage) in experimental.items():
        frequencies, spectrum, full_spectrum_peak = analyze_spectrum(times, voltage)
        uncertainty = MEASUREMENT_UNCERTAINTY_V / np.sqrt(voltage.size)
        experimental_spectra[name] = (
            frequencies,
            spectrum / full_spectrum_peak,
            np.full(frequencies.shape, uncertainty / full_spectrum_peak),
        )

    envelope_frequencies = np.linspace(80, 800, 800)
    envelope = (envelope_frequencies, FUNDAMENTAL_FREQUENCY_HZ / envelope_frequencies)
    plot_specs = (
        ("single", "Singlecoil"),
        ("parallel", "Humbucker_parra"),
        ("series", "Humbucker_serie"),
    )
    for key, filename in plot_specs:
        experimental_frequencies, experimental_spectrum, errors = experimental_spectra[
            key
        ]
        model_curve = (
            model_frequencies,
            model_spectra[key],
            THEORY_STYLES[1 if key == "single" else (0 if key == "parallel" else 2)],
        )
        measured = (
            experimental_frequencies,
            experimental_spectrum,
            {
                **EXPERIMENT_STYLES[
                    1 if key == "single" else (0 if key == "parallel" else 2)
                ],
                "yerr": errors,
            },
        )
        plot_spectrum(
            OUTPUT_DIRECTORY / "FFT" / f"{filename}_th.png", (model_curve,), envelope
        )
        plot_spectrum(
            OUTPUT_DIRECTORY / "FFT" / f"{filename}_exp.png", (measured,), envelope
        )
        plot_spectrum(
            OUTPUT_DIRECTORY / "FFT" / f"{filename}.png",
            (model_curve, measured),
            envelope,
        )

    plot_spatial_filtering(experimental_spectra)


def plot_spatial_filtering(
    experimental_spectra: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Compare measured harmonic attenuation with the humbucker comb-filter model."""
    single_frequencies, single_spectrum, _ = experimental_spectra["single"]
    series_frequencies, series_spectrum, _ = experimental_spectra["series"]
    _, single_peaks_a = model.extraire_pics(
        single_frequencies, single_spectrum, prominence=PEAK_PROMINENCE, distance=5
    )
    series_peaks_f, series_peaks_a = model.extraire_pics(
        series_frequencies, series_spectrum, prominence=PEAK_PROMINENCE, distance=5
    )

    model_frequencies = np.linspace(30, 20_000, 1_000)
    model_signal, comb_filter, single_signal = model.attenuation_theorique_db(
        model_frequencies, *SERIES_POSITIONS_M
    )
    model_attenuation = 20 * np.log10(
        np.abs(model_signal / (single_signal + 1e-6)) + 1e-6
    )
    shared_peak_count = min(single_peaks_a.size, series_peaks_a.size)
    corrected_frequencies = np.array([990, 1320, 1650, 1980, 2310, 2640, 2970, 3300])
    corrected_attenuation = np.array(
        [-5.2, -4, -10.5, -16.8, -15.5, -14.2, -20.4, -28.7]
    )
    measured_attenuation = 20 * np.log10(
        np.abs(
            series_peaks_a[:shared_peak_count]
            / (single_peaks_a[:shared_peak_count] + 1e-6)
        )
        + 1e-6
    )
    measured_frequencies = np.concatenate(
        (series_peaks_f[:shared_peak_count], corrected_frequencies)
    )
    measured_attenuation = np.concatenate((measured_attenuation, corrected_attenuation))

    model.save_plot(
        str(OUTPUT_DIRECTORY / "Filtrage_spatial.png"),
        (
            model_frequencies,
            model_attenuation,
            {"color": "#888888", "label": "Relative gain"},
        ),
        (measured_frequencies, measured_attenuation, EXPERIMENT_STYLES[2]),
        (
            model_frequencies,
            20 * np.log10(np.abs(comb_filter) + 1e-6),
            {"color": "#000000", "label": "Spatial comb filter", "linewidth": 3},
        ),
        title="Humbucker spatial filtering",
        xlabel="Frequency [Hz]",
        ylabel="Gain [dB]",
        xlim=(30, 20_000),
        vspan=True,
        log=True,
    )


def main() -> None:
    """Run mechanical-response analysis and save the spectrum plots."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    create_spectrum_plots()
    LOGGER.info("Mechanical-response figures saved to %s", OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
