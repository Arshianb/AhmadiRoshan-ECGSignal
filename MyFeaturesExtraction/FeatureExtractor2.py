from scipy.signal import find_peaks
import numpy as np
class periodicity_measurement:
    def __init__(self, ):
        self.name = "periodicity_measurement"
    def call(ecg_signal, fs):
        # Find the R peaks using the find_peaks function
        peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.5), height=0.5)

        # Calculate the R-R intervals
        rr_intervals = np.diff(peaks) / fs

        # Calculate the autocorrelation of the R-R intervals
        auto_corr = np.correlate(rr_intervals, rr_intervals, mode='full')

        # Find the index of the first maximum peak in the autocorrelation
        max_peak_index = np.argmax(auto_corr)

        # Calculate the periodicity measurement as the distance between the first maximum peak and the center of the autocorrelation
        periodicity = max_peak_index - len(rr_intervals) + 1

        # Return the periodicity measurement
        return periodicity
    

class sharpness_measurement:
    def __init__(self, ):
        self.name = "sharpness_measurement"
    def call(ecg_signal):
        # Calculate the gradient of the signal
        gradient = np.gradient(ecg_signal)

        # Calculate the sharpness measurement as the maximum absolute gradient value
        sharpness = np.max(np.abs(gradient))

        # Return the sharpness measurement
        return sharpness
    
class correlation_measurement:
    def __init__(self, ):
        self.name = "correlation_measurement"
    def call(ecg_signal1, ecg_signal2):
        # Calculate the Pearson correlation coefficient between the two signals
        corr_coef = np.corrcoef(ecg_signal1, ecg_signal2)[0, 1]

        # Return the correlation measurement
        return corr_coef
    
class peak_height_stability:
    def __init__(self, ):
        self.name = "peak_height_stability"
    def call(ecg_signal, fs):
        # Find the R peaks using the find_peaks function
        peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.5), height=0.5)

        # Calculate the height of each peak
        peak_heights = ecg_signal[peaks]

        # Calculate the coefficient of variation of the peak heights
        cv = np.std(peak_heights) / np.mean(peak_heights)

        # Return the peak height stability measurement
        return cv
    
class max_period_between_r_waves:
    def __init__(self, ):
        self.name = "max_period_between_r_waves"
    def call(ecg_signal, fs):
        # Find the R peaks using the find_peaks function
        peaks, _ = find_peaks(ecg_signal, distance=int(fs * 0.5), height=0.5)

        # Calculate the time difference between consecutive R peaks
        time_diff = np.diff(peaks) / fs

        # Return the maximum period between consecutive R waves
        return np.max(time_diff)
    

class histogram_analysis_mean:
    def __init__(self, ):
        self.name = "histogram_analysis_mean"
    def call(ecg_signal, num_bins=50):
        # Calculate the histogram of the ECG signal
        hist, bins = np.histogram(ecg_signal, bins=num_bins)

        # Calculate the midpoint of each bin
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate the weighted average of the bin centers using the histogram values as weights
        mean = np.average(bin_centers, weights=hist)

        return mean
    
class histogram_analysis_std_dev:
    def __init__(self, ):
        self.name = "histogram_analysis_std_dev"
    def call(ecg_signal, num_bins=50):
        # Calculate the histogram of the ECG signal
        hist, bins = np.histogram(ecg_signal, bins=num_bins)

        # Find the peaks in the histogram
        peaks, _ = find_peaks(hist)

        # Calculate the distance between adjacent peaks
        distances = np.diff(peaks)

        # Calculate the standard deviation of the distances
        std_dev = np.std(distances)

        return std_dev
    
class calc_blank_area_swing:
    def __init__(self, ):
        self.name = "calc_blank_area_swing"
    def call(signal):
        # Determine the blank areas (i.e., the areas of the signal that are near zero)
        blank_areas = np.abs(signal) < 0.1  # Change 0.1 to your desired threshold

        # Determine the lengths of each blank area
        blank_area_lengths = np.diff(np.where(np.concatenate(([blank_areas[0]], blank_areas[:-1] != blank_areas[1:], [True])))[0])[::2]

        # Calculate the maximum blank area swing
        max_blank_area_swing = np.max(blank_area_lengths) - np.min(blank_area_lengths)

        return max_blank_area_swing
    
class calc_median_neighborhood_swing:
    def __init__(self, ):
        self.name = "calc_median_neighborhood_swing"
    def call(signal, neighborhood_size=3):
        # Calculate the swing of each neighborhood
        neighborhood_swings = [np.max(signal[i:i+neighborhood_size]) - np.min(signal[i:i+neighborhood_size]) for i in range(len(signal) - neighborhood_size + 1)]
        
        # Calculate the median neighborhood swing
        median_neighborhood_swing = np.median(neighborhood_swings)
        
        return median_neighborhood_swing
    
class calc_blank_to_median_swing_ratio:
    def __init__(self, ):
        self.name = "calc_blank_to_median_swing_ratio"
    def call(signal):
        # Determine the blank areas (i.e., the areas of the signal that are near zero)
        blank_areas = np.abs(signal) < 0.1  # Change 0.1 to your desired threshold
        
        # Determine the lengths of each blank area
        blank_area_lengths = np.diff(np.where(np.concatenate(([blank_areas[0]], blank_areas[:-1] != blank_areas[1:], [True])))[0])[::2]
        
        # Calculate the maximum blank area swing
        max_blank_area_swing = np.max(blank_area_lengths) - np.min(blank_area_lengths)
        
        # Calculate the median neighborhood swing
        neighborhood_size = 3
        median_neighborhood_swing = np.median([np.max(signal[i:i+neighborhood_size]) - np.min(signal[i:i+neighborhood_size]) for i in range(len(signal) - neighborhood_size + 1)])
        
        # Calculate the "blank area swing to median neighborhood swing ratio"
        blank_to_median_swing_ratio = max_blank_area_swing / median_neighborhood_swing
        
        return blank_to_median_swing_ratio
    
class asystole_features:
    def __init__(self, ):
        self.name = "asystole_features"
    def call(rr_intervals):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute the time intervals between successive heart beats
        time_intervals = np.diff(np.arange(len(rr_intervals))*rr_intervals[0])
        
        # Compute the mean and standard deviation of heart rate
        mean_hr = np.mean(heart_rates)
        std_hr = np.std(heart_rates)
        
        # Compute the proportion of time with no heart beats (i.e., asystole)
        asystole_time = np.sum(time_intervals[heart_rates == 0])
        total_time = np.sum(time_intervals)
        asystole_prop = asystole_time / total_time
        
        # Compute the maximum and minimum heart rates during the asystole period
        asystole_idx = np.where(heart_rates == 0)[0]
        max_hr_asystole = 0
        min_hr_asystole = float('inf')
        for i in range(len(asystole_idx)-1):
            start_idx = asystole_idx[i]
            end_idx = asystole_idx[i+1]
            if end_idx - start_idx > 1:
                hr_asystole = 60 / rr_intervals[start_idx:end_idx]
                max_hr_asystole = max(max_hr_asystole, np.max(hr_asystole))
                min_hr_asystole = min(min_hr_asystole, np.min(hr_asystole))
        
        # Compute the proportion of time in different heart rate ranges
        hr_ranges = [0, 40, 60, 100, 120, float('inf')]
        hr_counts = [np.sum((heart_rates >= hr_ranges[i]) & (heart_rates < hr_ranges[i+1])) for i in range(len(hr_ranges)-1)]
        hr_props = np.array(hr_counts) / len(heart_rates)
        
        # Combine all features into a single feature vector
        asystole_features = [mean_hr, std_hr, asystole_prop, max_hr_asystole, min_hr_asystole] + list(hr_props)
        return asystole_features
    

class min_hr_4beats:
    def __init__(self, ):
        self.name = "min_hr_4beats"
    def call(rr_intervals):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Find the minimum heart rate across 4 beats
        min_hr = float('inf')
        for i in range(len(heart_rates) - 3):
            hr_4beats = np.mean(heart_rates[i:i+4])
            if hr_4beats < min_hr:
                min_hr = hr_4beats
        return min_hr
    
class num_beats_slower_than_46:
    def __init__(self, ):
        self.name = "num_beats_slower_than_46"
    def call(rr_intervals):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Count the number of beats slower than 46 b.p.m.
        count = 0
        for hr in heart_rates:
            if hr < 46:
                count += 1
        return count




import scipy.signal as signal


class max_mean_diff_LF_SUB_peaks:
    def __init__(self, ):
        self.name = "max_mean_diff_LF_SUB_peaks"
    def call(ecg_signal, sampling_rate):
        # Apply bandpass filter to ECG signal to isolate LF frequency band (0.04-0.15 Hz)
        b, a = signal.butter(4, [0.04, 0.15], btype='bandpass', fs=sampling_rate)
        ecg_lf = signal.filtfilt(b, a, ecg_signal)
        
        # Find peaks in LF signal
        lf_peaks, _ = signal.find_peaks(ecg_lf)
        
        # Compute mean differences between adjacent peaks
        lf_diffs = np.diff(ecg_lf[lf_peaks])
        
        # Compute mean of absolute differences between adjacent peaks
        lf_diffs_abs_mean = np.mean(np.abs(lf_diffs))
        
        # Compute maximum mean difference between adjacent sub-peaks
        sub_peak_diffs = np.split(lf_diffs, np.where(lf_diffs > lf_diffs_abs_mean)[0] + 1)
        sub_peak_diffs_means = [np.mean(np.abs(diff)) for diff in sub_peak_diffs if len(diff) > 1]
        
        if len(sub_peak_diffs_means) > 0:
            max_mean_diff = np.max(sub_peak_diffs_means)
        else:
            max_mean_diff = 0
        
        return max_mean_diff

import peakutils

# def calculate_R_peaks(signal):
#   return peakutils.indexes(signal, thres=0.5, min_dist=30)

# def detect_r_peaks(ecg_signal, fs=1000):
#     """
#     Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.
    
#     Parameters
#     ----------
#     ecg_signal : ndarray
#         ECG signal to analyze.
#     fs : int, optional
#         Sampling frequency of the ECG signal (in Hz). Default is 1000 Hz.
        
#     Returns
#     -------
#     r_peaks : ndarray
#         Array of R-peak indices in the ECG signal.
#     """
    
#     # Define filter parameters
#     f1 = 5       # Low-pass filter cutoff frequency
#     f2 = 15      # High-pass filter cutoff frequency
#     N = 201      # Filter order

#     # Design bandpass filter using FIR window method
#     h = np.zeros(N)
#     for n in range(N):
#         if n == (N-1)/2:
#             h[n] = (2 * (f2/fs - f1/fs))
#         else:
#             h[n] = (np.sin(2*np.pi*f2*(n-(N-1)/2)/fs) - np.sin(2*np.pi*f1*(n-(N-1)/2)/fs)) / (np.pi*(n-(N-1)/2))

#     # Apply filter to ECG signal
#     filtered_ecg = np.convolve(ecg_signal, h)

#     # Derive squared ECG signal for QRS detection
#     squared_ecg = filtered_ecg ** 2

#     # Define moving average window for integration
#     win_size = int(0.150 * fs)    # 150 ms window
#     window = np.ones(win_size) / win_size

#     # Apply moving average filter to squared ECG signal
#     smoothed_ecg = np.convolve(squared_ecg, window)

#     # Find peaks in smoothed ECG signal
#     peak_indices = []
#     for i in range(win_size, len(smoothed_ecg)):
#         if smoothed_ecg[i] > np.max(smoothed_ecg[i-win_size:i]) and smoothed_ecg[i] > 0.5*np.max(smoothed_ecg):
#             peak_indices.append(i)

#     # Return array of R-peak indices
#     r_peaks = np.array(peak_indices)
#     return r_peaks

import biosppy.signals.ecg as ecg

class detect_rpeaks:
    def __init__(self, ):
        self.name = "detect_rpeaks"
    def call(ecg_signal, fs):
        """
        Detect R-peaks in an ECG signal using the Christov method.

        Args:
            ecg_signal (array): 1D numpy array containing the ECG signal.
            fs (float): Sampling rate of the ECG signal in Hz.

        Returns:
            array: 1D numpy array containing the indices of the R-peaks in the ECG signal.
        """
        # Compute R-peaks
        out = ecg.christov_segmenter(signal=ecg_signal, sampling_rate=fs)
        rpeaks = out['rpeaks']

        return rpeaks

class calculate_QRS_duration:
    def __init__(self, ):
        self.name = "calculate_QRS_duration"
    def call(signal, r_peaks):
        q_peak = np.argmin(signal[r_peaks[0]:r_peaks[1]]) + r_peaks[0]
        s_peak = np.argmin(signal[r_peaks[1]:r_peaks[2]]) + r_peaks[1]

        # Calculate QRS duration
        qrs_duration = s_peak - q_peak
        return qrs_duration

class calculate_Heart_rate:
    def __init__(self, ):
        self.name = "calculate_Heart_rate"
    def call(r_peaks):
        return 60 / np.mean(np.diff(r_peaks))

class calculate_QT_interval:
    def __init__(self, ):
        self.name = "calculate_QT_interval"
    def call(signal, r_peaks):
        q_onset = np.argmin(signal[r_peaks[0]-30:r_peaks[0]]) + r_peaks[0] - 30
        t_end = np.argmax(signal[r_peaks[1]:r_peaks[2]]) + r_peaks[1]

        # Calculate QT interval
        qt_interval = t_end - q_onset
        return qt_interval


class max_hr_over_5_beats:
    def __init__(self, ):
        self.name = "max_hr_over_5_beats"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute rolling 5-beat maximum heart rate
        window_size = 5
        max_heart_rates = np.zeros(len(heart_rates) - window_size + 1)
        for i in range(len(max_heart_rates)):
            max_heart_rates[i] = np.max(heart_rates[i:i+window_size])
        
        # Return maximum rolling 5-beat heart rate
        max_hr_over_5_beats = np.max(max_heart_rates)
        return max_hr_over_5_beats


class corr_measure_over_5_beats:
    def __init__(self, ):
        self.name = "corr_measure_over_5_beats"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute correlation between HRV of 5 consecutive beats
        window_size = 5
        corr_coeffs = np.zeros(len(hrv) - window_size + 1)
        for i in range(len(corr_coeffs)):
            corr_coeffs[i] = np.corrcoef(hrv[i:i+window_size])[0, 1]
        
        # Return mean correlation coefficient
        corr_measure_over_5_beats = np.mean(corr_coeffs)
        return corr_measure_over_5_beats


class sharpness_measure_over_5_beats:
    def __init__(self, ):
        self.name = "sharpness_measure_over_5_beats"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Find peak frequency within 0.04-0.15 Hz range
        freq_range = (f >= 0.04) & (f <= 0.15)
        peak_freq = f[freq_range][np.argmax(psd[freq_range])]
        
        # Compute sharpness measure as PSD at peak frequency
        sharpness_measure_over_5_beats = psd[f == peak_freq][0]
        return sharpness_measure_over_5_beats



class five_consecutive_vt_beats:
    def __init__(self, ):
        self.name = "five_consecutive_vt_beats"
    def call(rr_intervals, vt_threshold=100):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Detect VT beats above threshold
        vt_beats = heart_rates > vt_threshold
        
        # Compute run lengths of VT beats
        vt_runs = np.diff(np.where(np.concatenate(([vt_beats[0]],
                                    vt_beats[:-1] != vt_beats[1:],
                                    [True])))[0])[::2]
        
        # Check if any run length is greater than or equal to 5
        five_consecutive_vt_beats = any(vt_runs >= 5)
        return five_consecutive_vt_beats


class number_of_peaks_above_threshold:
    def __init__(self, ):
        self.name = "number_of_peaks_above_threshold"
    def call(rr_intervals, sampling_rate, power_threshold=0.2):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Count number of peaks above threshold
        peaks = signal.find_peaks(psd_norm)[0]
        num_peaks_above_threshold = np.sum(psd_norm[peaks] > power_threshold)
        return num_peaks_above_threshold

class max_power_to_total_power_ratio:
    def __init__(self, ):
        self.name = "max_power_to_total_power_ratio"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Compute ratio of max power to total power in PSD
        max_power_to_total_power_ratio = np.max(psd) / np.sum(psd)
        return max_power_to_total_power_ratio



# threshold=0.2
class count_peaks_with_normalized_power_above:
    def __init__(self, ):
        self.name = "count_peaks_with_normalized_power_above"
    def call(signal, threshold):
        """Counts the number of peaks in a signal with normalized power above a threshold.

        Args:
            signal: The signal to be analyzed.
            threshold: The threshold for normalized power.

        Returns:
            The number of peaks in the signal with normalized power above the threshold.
        """

        # Calculate the Fourier transform of the signal.
        fft_signal = np.fft.fft(signal)

        # Calculate the magnitudes of the Fourier transform.
        magnitudes = np.abs(fft_signal)

        # Calculate the normalized powers.
        normalized_powers = magnitudes / np.sum(magnitudes)

        # Count the number of peaks with normalized power above the threshold.
        count = np.count_nonzero(normalized_powers > threshold)

        # Return the number of peaks.
        return count


class count_consecutive_vt_beats_above:
    def __init__(self, ):
        self.name = "count_consecutive_vt_beats_above"
    def call(signal, rate):
        """Counts the number of consecutive VT beats in a signal with a rate above a threshold.

        Args:
            signal: The signal to be analyzed.
            rate: The threshold for rate.

        Returns:
            The number of consecutive VT beats in the signal with a rate above the threshold.
        """

        # Calculate the number of beats in the signal.
        number_of_beats = len(signal)

        # Initialize the count of consecutive VT beats.
        count = 0

        # Iterate through the signal.
        for i in range(number_of_beats):
            # If the current beat is a VT beat and the previous beat is also a VT beat, increment the count.
            if signal[i] > rate and (i == 0 or signal[i - 1] > rate):
                count += 1
            else:
                count = 0

        # Return the count of consecutive VT beats.
        return count


class median_frequency:
    def __init__(self, ):
        self.name = "median_frequency"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Compute cumulative sum of PSD
        psd_cumsum = np.cumsum(psd) / np.sum(psd)
        
        # Find frequency bin at which cumulative sum reaches 0.5 (i.e., median frequency)
        median_freq_bin = np.argmax(psd_cumsum >= 0.5)
        
        # Compute median frequency
        median_freq = f[median_freq_bin]
        return median_freq

class mean_frequency:
    def __init__(self, ):
        self.name = "mean_frequency"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Compute mean frequency
        mean_freq = np.sum(psd * f) / np.sum(psd)
        return mean_freq


class dominant_frequency:
    def __init__(self, ):
        self.name = "dominant_frequency"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Find frequency bin with highest power
        max_power_bin = np.argmax(psd)
        
        # Compute dominant frequency
        dominant_freq = f[max_power_bin]
        return dominant_freq


class bandwidth:
    def __init__(self, ):
        self.name = "bandwidth"
    def call(rr_intervals, sampling_rate):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute power spectral density (PSD) of HRV signal
        f, psd = signal.periodogram(hrv, fs=sampling_rate)
        
        # Compute spectral bandwidth
        norm_psd = psd / np.sum(psd) # normalize PSD
        mean_freq = np.sum(f * norm_psd) # compute mean frequency
        variance = np.sum((f - mean_freq)**2 * norm_psd) # compute variance
        bw = 2 * np.sqrt(variance) # compute bandwidth
        return bw


from pyentrp import entropy as ent

class complexity_measure:
    def __init__(self, ):
        self.name = "complexity_measure"
    def call(rr_intervals):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Compute HRV from heart rates
        hrv = np.diff(heart_rates)
        
        # Compute sample entropy of HRV signal
        sampen = ent.sample_entropy(hrv)
        return sampen


class num_heartbeats:
    def __init__(self, ):
        self.name = "num_heartbeats"
    def call(ecg_signal, window_size, sampling_rate):
        # Compute the number of heartbeats within the window of analysis
        num_beats = 0
        window_samples = int(window_size * sampling_rate)
        for i in range(0, len(ecg_signal)-window_samples, window_samples):
            window = ecg_signal[i:i+window_samples]
            peaks, _ = signal.find_peaks(window, distance=sampling_rate/2)
            num_beats += len(peaks)
        return num_beats

class max_hr_17beats:
    def __init__(self, ):
        self.name = "max_hr_17beats"
    def call(rr_intervals):
        # Compute heart rates from RR intervals
        heart_rates = 60 / rr_intervals
        
        # Find the maximum heart rate across 17 beats
        max_hr = 0
        for i in range(len(heart_rates) - 16):
            hr_17beats = np.mean(heart_rates[i:i+17])
            if hr_17beats > max_hr:
                max_hr = hr_17beats
        return max_hr
