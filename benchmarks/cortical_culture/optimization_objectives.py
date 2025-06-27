"""
Optimization objectives for fitting neural simulations to experimental MEA data.

This module extracts key features from experimental neural recordings that can be used
as objectives for black-box optimization to fit computational models to the data.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import json


class NeuralObjectiveExtractor:
    """Extract optimization objectives from neural recording data."""

    def __init__(self, experimental_data_path: str):
        """
        Initialize with experimental data.

        Args:
            experimental_data_path: Path to the experimental recording pickle file
        """
        self.experimental_data_path = experimental_data_path
        self.frequency_bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 100),
        }
        self.load_experimental_data()
        self.extract_experimental_features()

    def load_experimental_data(self):
        """Load experimental data from pickle file."""
        with open(self.experimental_data_path, "rb") as f:
            recording = pickle.load(f)

        self.exp_data = recording["data"]  # (n_samples, n_channels)
        self.exp_timestamps = recording["timestamps"]  # (n_samples,)
        self.exp_fs = recording["rate"]

        print(f"Loaded experimental data:")
        print(f"  Shape: {self.exp_data.shape}")
        print(f"  Sampling rate: {self.exp_fs} Hz")
        print(f"  Duration: {len(self.exp_timestamps)/self.exp_fs:.2f} seconds")

    def extract_experimental_features(self):
        """Extract reference features from experimental data."""
        # Use representative channel (channel 4 as in original analysis)
        channel_idx = 4
        self.exp_channel_data = self.exp_data[:, channel_idx]

        # Compute power spectral density
        nperseg = int(self.exp_fs * 0.5)  # 0.5 second windows
        self.exp_freqs, self.exp_psd = signal.welch(
            self.exp_channel_data, self.exp_fs, nperseg=nperseg
        )

        # Extract frequency band powers
        self.exp_band_powers = {}
        for band_name, (low, high) in self.frequency_bands.items():
            if high <= self.exp_fs / 2:
                mask = (self.exp_freqs >= low) & (self.exp_freqs <= high)
                self.exp_band_powers[band_name] = np.mean(self.exp_psd[mask])

        # Extract additional spectral features
        self.exp_peak_freq = self.exp_freqs[np.argmax(self.exp_psd)]
        self.exp_total_power = np.sum(self.exp_psd)
        self.exp_spectral_centroid = np.sum(self.exp_freqs * self.exp_psd) / np.sum(
            self.exp_psd
        )

        # Frequency range for analysis (0-100 Hz)
        freq_mask = self.exp_freqs <= 100
        self.exp_freqs_limited = self.exp_freqs[freq_mask]
        self.exp_psd_limited = self.exp_psd[freq_mask]

        print(f"Extracted experimental features:")
        print(f"  Peak frequency: {self.exp_peak_freq:.1f} Hz")
        print(f"  Total power: {self.exp_total_power:.2e}")
        print(f"  Spectral centroid: {self.exp_spectral_centroid:.1f} Hz")
        for band, power in self.exp_band_powers.items():
            print(f"  {band} power: {power:.2e}")

    def compute_psd_features(self, data: np.ndarray, fs: float) -> Dict:
        """
        Compute power spectral density features from neural data.

        Args:
            data: Neural data array (n_samples,) or (n_samples, n_channels)
            fs: Sampling frequency

        Returns:
            Dictionary of PSD features
        """
        # If multi-channel, use first channel or average
        if data.ndim > 1:
            if data.shape[1] > 1:
                # Use the same channel as experimental data for consistency
                signal_data = data[:, min(4, data.shape[1] - 1)]
            else:
                signal_data = data[:, 0]
        else:
            signal_data = data

        # Compute power spectral density
        nperseg = int(fs * 0.5)  # 0.5 second windows
        freqs, psd = signal.welch(signal_data, fs, nperseg=nperseg)

        # Limit to 0-100 Hz range
        freq_mask = freqs <= 100
        freqs_limited = freqs[freq_mask]
        psd_limited = psd[freq_mask]

        features = {}

        # Frequency band powers
        for band_name, (low, high) in self.frequency_bands.items():
            if high <= fs / 2:
                mask = (freqs >= low) & (freqs <= high)
                features[f"{band_name.lower()}_power"] = np.mean(psd[mask])

        # Peak frequency
        features["peak_frequency"] = freqs[np.argmax(psd)]

        # Total power
        features["total_power"] = np.sum(psd)

        # Spectral centroid
        features["spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd)

        # Power slope (1/f noise characteristic)
        # Fit log-log relationship in 1-50 Hz range
        slope_mask = (freqs >= 1) & (freqs <= 50)
        if np.sum(slope_mask) > 10:  # Need enough points for fit
            log_freqs = np.log10(freqs[slope_mask])
            log_psd = np.log10(psd[slope_mask] + 1e-12)
            slope, _ = np.polyfit(log_freqs, log_psd, 1)
            features["power_slope"] = slope
        else:
            features["power_slope"] = 0

        # Spectral entropy
        psd_norm = psd_limited / np.sum(psd_limited)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm))

        # Relative band powers (normalized by total power)
        total_power = np.sum(psd)
        for band_name in self.frequency_bands.keys():
            if f"{band_name.lower()}_power" in features:
                features[f"{band_name.lower()}_relative_power"] = (
                    features[f"{band_name.lower()}_power"] / total_power
                )

        return features, freqs_limited, psd_limited

    def compute_objectives(self, sim_data: np.ndarray, sim_fs: float) -> Dict:
        """
        Compute optimization objectives by comparing simulation to experimental data.

        Args:
            sim_data: Simulated neural data array
            sim_fs: Sampling frequency of simulation

        Returns:
            Dictionary of objective values (lower is better)
        """
        # Extract features from simulation
        sim_features, sim_freqs, sim_psd = self.compute_psd_features(sim_data, sim_fs)

        objectives = {}

        # 1. Frequency band power matching
        for band_name in self.frequency_bands.keys():
            exp_power = self.exp_band_powers[band_name]
            sim_power = sim_features[f"{band_name.lower()}_power"]

            # Relative error in log space (since powers span many orders of magnitude)
            rel_error = abs(np.log10(sim_power + 1e-12) - np.log10(exp_power + 1e-12))
            objectives[f"{band_name.lower()}_band_error"] = rel_error

        # 2. Peak frequency matching
        peak_freq_error = (
            abs(sim_features["peak_frequency"] - self.exp_peak_freq)
            / self.exp_peak_freq
        )
        objectives["peak_frequency_error"] = peak_freq_error

        # 3. Spectral centroid matching
        centroid_error = (
            abs(sim_features["spectral_centroid"] - self.exp_spectral_centroid)
            / self.exp_spectral_centroid
        )
        objectives["spectral_centroid_error"] = centroid_error

        # 4. Power spectrum shape correlation
        # Interpolate simulation PSD to experimental frequency grid
        sim_psd_interp = np.interp(self.exp_freqs_limited, sim_freqs, sim_psd)

        # Compute correlation in log space
        log_exp_psd = np.log10(self.exp_psd_limited + 1e-12)
        log_sim_psd = np.log10(sim_psd_interp + 1e-12)

        correlation, _ = pearsonr(log_exp_psd, log_sim_psd)
        objectives["psd_correlation_error"] = 1 - correlation  # Convert to minimization

        # 5. Power slope matching (1/f noise)
        if "power_slope" in sim_features:
            slope_error = abs(
                sim_features["power_slope"] - getattr(self, "exp_power_slope", -1)
            )  # Default -1 for 1/f noise
            objectives["power_slope_error"] = slope_error

        # 6. Spectral entropy matching
        entropy_error = abs(
            sim_features["spectral_entropy"] - getattr(self, "exp_spectral_entropy", 5)
        )  # Reasonable default
        objectives["spectral_entropy_error"] = entropy_error

        # 7. Relative power distribution matching
        for band_name in self.frequency_bands.keys():
            if f"{band_name.lower()}_relative_power" in sim_features:
                exp_rel_power = self.exp_band_powers[band_name] / self.exp_total_power
                sim_rel_power = sim_features[f"{band_name.lower()}_relative_power"]
                rel_power_error = abs(sim_rel_power - exp_rel_power)
                objectives[f"{band_name.lower()}_relative_power_error"] = (
                    rel_power_error
                )

        return objectives

    def get_objective_weights(self) -> Dict[str, float]:
        """
        Get recommended weights for multi-objective optimization.

        Returns:
            Dictionary of objective weights
        """
        weights = {
            # Frequency band errors (high importance)
            "delta_band_error": 2.0,
            "theta_band_error": 2.0,
            "alpha_band_error": 2.0,
            "beta_band_error": 2.0,
            "gamma_band_error": 2.0,
            # Overall spectral shape (very high importance)
            "psd_correlation_error": 5.0,
            # Key spectral features (medium importance)
            "peak_frequency_error": 1.5,
            "spectral_centroid_error": 1.5,
            # Fine-grained features (lower importance)
            "power_slope_error": 1.0,
            "spectral_entropy_error": 1.0,
            # Relative power distribution (medium importance)
            "delta_relative_power_error": 1.0,
            "theta_relative_power_error": 1.0,
            "alpha_relative_power_error": 1.0,
            "beta_relative_power_error": 1.0,
            "gamma_relative_power_error": 1.0,
        }
        return weights

    def save_experimental_targets(self, filepath: str):
        """Save experimental targets to JSON file for reference."""
        targets = {
            "experimental_data_path": self.experimental_data_path,
            "sampling_rate": float(self.exp_fs),
            "peak_frequency": float(self.exp_peak_freq),
            "total_power": float(self.exp_total_power),
            "spectral_centroid": float(self.exp_spectral_centroid),
            "band_powers": {k: float(v) for k, v in self.exp_band_powers.items()},
            "frequency_bands": self.frequency_bands,
        }

        with open(filepath, "w") as f:
            json.dump(targets, f, indent=2)

        print(f"Experimental targets saved to {filepath}")

    def visualize_comparison(
        self, sim_data: np.ndarray, sim_fs: float, save_path: Optional[str] = None
    ):
        """
        Visualize comparison between experimental and simulated data.

        Args:
            sim_data: Simulated neural data
            sim_fs: Sampling frequency of simulation
            save_path: Optional path to save the figure
        """
        sim_features, sim_freqs, sim_psd = self.compute_psd_features(sim_data, sim_fs)
        objectives = self.compute_objectives(sim_data, sim_fs)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Experimental vs Simulated Neural Data Comparison", fontsize=16)

        # 1. Power spectral density comparison
        axes[0, 0].loglog(
            self.exp_freqs_limited,
            self.exp_psd_limited,
            "b-",
            label="Experimental",
            linewidth=2,
        )
        axes[0, 0].loglog(sim_freqs, sim_psd, "r--", label="Simulated", linewidth=2)
        axes[0, 0].set_xlabel("Frequency (Hz)")
        axes[0, 0].set_ylabel("Power Spectral Density")
        axes[0, 0].set_title("Power Spectral Density Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0.5, 100])

        # 2. Frequency band powers comparison
        bands = list(self.frequency_bands.keys())
        exp_powers = [self.exp_band_powers[band] for band in bands]
        sim_powers = [sim_features[f"{band.lower()}_power"] for band in bands]

        x_pos = np.arange(len(bands))
        width = 0.35

        axes[0, 1].bar(
            x_pos - width / 2, exp_powers, width, label="Experimental", alpha=0.7
        )
        axes[0, 1].bar(
            x_pos + width / 2, sim_powers, width, label="Simulated", alpha=0.7
        )
        axes[0, 1].set_xlabel("Frequency Bands")
        axes[0, 1].set_ylabel("Average Power")
        axes[0, 1].set_title("Frequency Band Power Comparison")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(bands)
        axes[0, 1].legend()
        axes[0, 1].set_yscale("log")

        # 3. Objective values
        obj_names = list(objectives.keys())[:10]  # Show first 10 objectives
        obj_values = [objectives[name] for name in obj_names]

        axes[1, 0].bar(range(len(obj_names)), obj_values)
        axes[1, 0].set_xlabel("Objective")
        axes[1, 0].set_ylabel("Error Value")
        axes[1, 0].set_title("Optimization Objectives (Lower is Better)")
        axes[1, 0].set_xticks(range(len(obj_names)))
        axes[1, 0].set_xticklabels(obj_names, rotation=45, ha="right")

        # 4. Key metrics comparison
        metrics = ["Peak Freq (Hz)", "Spectral Centroid (Hz)", "Total Power"]
        exp_metrics = [
            self.exp_peak_freq,
            self.exp_spectral_centroid,
            self.exp_total_power,
        ]
        sim_metrics = [
            sim_features["peak_frequency"],
            sim_features["spectral_centroid"],
            sim_features["total_power"],
        ]

        # Normalize for comparison
        exp_metrics_norm = np.array(exp_metrics) / np.array(exp_metrics)
        sim_metrics_norm = np.array(sim_metrics) / np.array(exp_metrics)

        x_pos = np.arange(len(metrics))
        axes[1, 1].bar(
            x_pos - width / 2, exp_metrics_norm, width, label="Experimental", alpha=0.7
        )
        axes[1, 1].bar(
            x_pos + width / 2, sim_metrics_norm, width, label="Simulated", alpha=0.7
        )
        axes[1, 1].set_xlabel("Metrics")
        axes[1, 1].set_ylabel("Normalized Value")
        axes[1, 1].set_title("Key Metrics Comparison (Normalized)")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].axhline(y=1, color="k", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        # Print summary
        print(f"\nComparison Summary:")
        print(f"PSD Correlation: {1 - objectives['psd_correlation_error']:.3f}")
        print(f"Peak Frequency Error: {objectives['peak_frequency_error']:.3f}")
        print(f"Total Objective Score: {sum(objectives.values()):.3f}")


def create_example_optimization_function(experimental_data_path: str):
    """
    Create an example optimization function that can be used with your optimizer.

    Args:
        experimental_data_path: Path to experimental data pickle file

    Returns:
        Function that takes simulation parameters and returns objectives
    """
    extractor = NeuralObjectiveExtractor(experimental_data_path)

    def objective_function(
        simulation_output: np.ndarray, simulation_fs: float = None
    ) -> np.ndarray:
        """
        Objective function for optimization.

        Args:
            simulation_output: Neural simulation output data
            simulation_fs: Sampling frequency (if None, use experimental fs)

        Returns:
            Array of objective values to minimize
        """
        if simulation_fs is None:
            simulation_fs = extractor.exp_fs

        objectives = extractor.compute_objectives(simulation_output, simulation_fs)
        weights = extractor.get_objective_weights()

        # Return weighted objectives as array
        objective_array = []
        for obj_name, obj_value in objectives.items():
            weight = weights.get(obj_name, 1.0)
            objective_array.append(weight * obj_value)

        return np.array(objective_array)

    return objective_function, extractor


if __name__ == "__main__":
    # Example usage
    experimental_data_path = "/home/frithjof/Scratch/data/miv-recording_0000.pkl"

    # Create objective extractor
    extractor = NeuralObjectiveExtractor(experimental_data_path)

    # Save experimental targets
    extractor.save_experimental_targets("experimental_targets.json")

    # Example: Test with synthetic data
    print("\nTesting with synthetic data...")

    # Generate synthetic neural-like data for testing
    t = np.linspace(0, 10, int(10 * extractor.exp_fs))
    synthetic_data = (
        0.1 * np.random.randn(len(t))  # Background noise
        + 0.05 * np.sin(2 * np.pi * 10 * t)  # Alpha-like oscillation
        + 0.02 * np.sin(2 * np.pi * 30 * t)  # Beta-like oscillation
        + 0.01 * np.sin(2 * np.pi * 60 * t)  # Gamma-like oscillation
    )

    # Compute objectives
    objectives = extractor.compute_objectives(synthetic_data, extractor.exp_fs)

    print("\nObjectives for synthetic data:")
    for obj_name, obj_value in objectives.items():
        print(f"  {obj_name}: {obj_value:.4f}")

    # Visualize comparison
    extractor.visualize_comparison(
        synthetic_data, extractor.exp_fs, "synthetic_comparison.png"
    )

    print(f"\nObjective extractor ready for optimization!")
    print(
        f"Use extractor.compute_objectives(sim_data, sim_fs) in your optimization loop."
    )
