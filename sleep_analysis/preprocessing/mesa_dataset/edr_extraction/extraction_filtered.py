"""Functions and Classes for EDR extraction applying filters on ECG data."""

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.stats
from scipy import signal

from sleep_analysis.preprocessing.mesa_dataset.edr_extraction.base_extraction import BaseExtraction


class ExtractionWavelet(BaseExtraction):
    """Abstract Class for Wavelet based algorithms.

    implements get_wavelet_coefficients(...) to unify the wavelet transform.
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        pass

    @staticmethod
    def get_wavelet_coefficients(ecg_signal: pd.DataFrame, sampling_rate: float):
        """Calculate the wavelet coefficients for the given ecg signal."""
        # Determine frequencies of interest
        lowest_frequency = 0.2
        highest_frequency = 3.8
        step_size = 0.01
        w = 5
        freqs = np.arange(start=lowest_frequency, stop=highest_frequency, step=step_size)
        # Calculate scales based of frequencies
        scales = w * sampling_rate / (2 * freqs * np.pi)

        coefficients = signal.cwt(
            data=ecg_signal.to_numpy(copy=True).ravel(),
            wavelet=signal.morlet2,
            widths=scales,
            w=5,
        )
        return coefficients, freqs


class ExtractionLindeberg(BaseExtraction):
    """Abstract Class for Lindeberg based algorithms.

    Band-pass filter between plausible respiratory frequencies
    from Charlton et al. http://dx.doi.org/10.1088/0967-3334/37/4/610
    based on the paper from Lindeberg et al.
    https://doi.org/10.1007/BF02457833
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        edr = nk.signal_filter(
            ecg_signal.iloc[:, 0],
            sampling_rate=sampling_rate,
            lowcut=0.066,
            highcut=1.0,
            order=10,
        )
        self.respiratory_signal = pd.DataFrame(data=edr, index=ecg_signal.index)
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionAddisonAM(ExtractionWavelet):
    """Addison  algorithm for EDR extraction (Amplitude Modulation).

    The maximum amplitude of the Continuous Wavelet Transform (CWT)
    within plausible cardiac frequencies (30–220 beats per minute)
    based on Ridges described in https://doi.org/10.1142/S0219691304000329
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Get Wavelet Coefficients
        coefficients, freqs = ExtractionAddisonAM.get_wavelet_coefficients(
            ecg_signal=ecg_signal, sampling_rate=sampling_rate
        )

        # filter frequencies between 30 and 220 bpm
        filter_range = (30, 220)
        rel_rows = [bool(filter_range[0] / 60 <= freq <= filter_range[1] / 60) for freq in freqs]
        filtered_coefficients = coefficients[rel_rows]

        # Get rid of 3rd dimension
        filtered_coefficients = np.reshape(filtered_coefficients, filtered_coefficients.shape[:2])

        # Transpose so its easier to iterate over time
        filtered_coefficients = filtered_coefficients.transpose()

        # Find maxima magnitudes and their indices
        max_magnitudes = [np.amax(np.abs(time_row)) for time_row in filtered_coefficients]

        # Saving with time index
        self.respiratory_signal = pd.DataFrame(data=max_magnitudes, index=ecg_signal.index)

        # Min/Max normalize
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionAddisonFM(ExtractionWavelet):
    """Addison  algorithm for EDR extraction (Frequency Modulation).

    The frequency indices of the maximum amplitudes over time of the Continuous Wavelet Transform (CWT)
    within plausible cardiac frequencies (30–220 beats per minute)
    based on ridges in https://doi.org/10.1142/S0219691304000329
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Get Wavelet Coefficients
        coefficients, freqs = ExtractionAddisonAM.get_wavelet_coefficients(
            ecg_signal=ecg_signal, sampling_rate=sampling_rate
        )

        # filter frequencies between 30 and 220 bpm
        filter_range = (30, 220)
        rel_rows = [bool(filter_range[0] / 60 <= freq <= filter_range[1] / 60) for freq in freqs]

        # Filter coefficients and frequencies
        filtered_coefficients = coefficients[rel_rows]
        filtered_frequencies = freqs[rel_rows]

        # Get rid of 3rd dimension
        filtered_coefficients = np.reshape(filtered_coefficients, filtered_coefficients.shape[:2])

        # Transpose so its easier to iterate over time
        filtered_coefficients = filtered_coefficients.transpose()

        # Find maxima magnitudes and their indices
        max_indices = [np.argmax(np.abs(time_row)) for time_row in filtered_coefficients]

        # take the maximum frequencies using the found indices (#Could maybe be optimised in one step)
        max_frequencies = [filtered_frequencies[i] for i in max_indices]

        # Save result as Dataframe
        self.respiratory_signal = pd.DataFrame(data=max_frequencies, index=ecg_signal.index)

        # Normalize and return
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionGarde(BaseExtraction):
    """Garde algorithm for EDR extraction.

    Filter using the centred-correntropy function (CCF) from Garde et al.
    https://doi.org/10.1371/journal.pone.0086427
    WARNING: Doesnt seem to work yet on 250Hz ECG Signals.
    Maybe too much noise but more likely #an implementation error in __ccf()
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # execute the ccf and normalize the result
        self.respiratory_signal = pd.DataFrame(self.__ccf(respiratory_signal=ecg_signal, sampling_rate=sampling_rate))
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self

    def __ccf(self, respiratory_signal: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
        """Centered correntropy function"""
        N = len(respiratory_signal.index)  # faster than taking the whole dataframe
        sigma = self.__calc_sigma(respiratory_signal=respiratory_signal)

        distances = np.empty(shape=(N, N), dtype=float)
        distances.fill(np.nan)
        # Calculate X(n) - X(n-m) -> L-upper Matrice (N x N)
        np_signal = np.array(respiratory_signal)
        # Begin at 1, because for 0 it results only in a empty row
        for m in range(0, N):
            differences = np_signal[m:N] - np_signal[0 : N - m]
            differences.resize((N,), refcheck=False)
            distances[m] = np.copy(differences)

        # Kernel density estimates as Matrix
        kdes = self.__gaussian_kernel(distances=distances, sigma=sigma)

        # find mean kernel density estimates V(m)
        summed_kdes = np.empty((N,))
        summed_kdes.fill(np.nan)
        for m in range(0, N):
            factor = 1 / (N - m + 1)
            sum = np.sum(kdes[m, :])
            summed_kdes[m] = factor * sum

        V = summed_kdes
        # Correntropy mean V_bar
        V_bar = (1 / (N**2)) * np.sum(kdes)
        # Centered CCF (Substract mean)
        V_c = V - V_bar
        return pd.DataFrame(data=V_c, index=respiratory_signal.index)

    def __calc_sigma(self, respiratory_signal: pd.DataFrame):
        N = len(respiratory_signal)
        interqr = scipy.stats.iqr(respiratory_signal)
        std = float(respiratory_signal.std())
        A = min([1.34 * interqr, std])
        sigma = 0.9 * A * (N ** (-1 / 5))
        return sigma

    def __gaussian_kernel(self, distances: np.array, sigma: float):
        kappa = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp((-1) * (distances**2) / (2 * (sigma**2)))
        return kappa


class ExtractionResampledAddisonFM(ExtractionWavelet):
    """Resampled Addison algorithm for EDR extraction (Frequency Modulation).

    The maximum amplitude of the Continuous Wavelet Transform (CWT)
    within plausible cardiac frequencies (30–220 beats per minute)
    with resamplimng to 25 Hz This was only for experimenting with sampling rate as CWT is
    very computational expensive the more timesteps are used.
    Did seem to produce simialar results to the not resampling method
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        # Resample ecg respiratory_signal to 25 Hz
        factor = 10
        ecg_signal = pd.DataFrame(
            data=scipy.signal.decimate(ecg_signal.iloc[:], factor, axis=0),
        )
        sampling_rate = sampling_rate / factor

        # Get Wavelet Coefficients
        (
            coefficients,
            frequencies,
        ) = ExtractionResampledAddisonFM.get_wavelet_coefficients(ecg_signal=ecg_signal, sampling_rate=sampling_rate)

        # filter frequencies between 30 and 220 bpm
        filter_range = (30, 220)
        rel_rows = [bool(freq >= filter_range[0] / 60 and freq <= filter_range[1] / 60) for freq in frequencies]
        # Filter to only relevant rows
        filtered_coefficients = coefficients[rel_rows]
        filtered_coefficients = np.reshape(filtered_coefficients, filtered_coefficients.shape[:2])
        # Transpose so its easier to iterate
        filtered_coefficients = filtered_coefficients.transpose()

        # Find maxima magnitudes and their indices
        max_magnitudes = [np.amax(np.abs(time_row)) for time_row in filtered_coefficients]
        max_indices = [np.argmax(np.abs(time_row)) for time_row in filtered_coefficients]

        # Saving with time index
        self.respiratory_signal = pd.DataFrame(data=max_indices, index=ecg_signal.index)

        # Normalize
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self
