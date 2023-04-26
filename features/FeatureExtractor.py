
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
import numpy as np
from features.full_waveform_features import *
import pandas as pd
import matplotlib.pyplot as plt
class Features:
    def __init__(self,feature_groups, labels=None):
        # self.fs = fs
        self.feature_groups = feature_groups
        self.labels = labels
        self.features = None
    def _preprocess_signal(self, fs, signal_raw, filter_bandwidth, normalize, polarity_check,
                           template_before, template_after):
        self.fs = fs
        self.filter_bandwidth = filter_bandwidth
        self.normalize=normalize
        self.polarity_check=polarity_check
        self.template_before=template_before
        self.template_after=template_after
        # Filter signal
        signal_filtered = self._apply_filter(signal_raw, filter_bandwidth)
        # plt.figure()
        # start = 0
        # end = 400
        # plt.plot(range(len(signal_filtered[start:end])), signal_filtered[start:end])
        # plt.figure()
        # plt.plot(range(len(signal_raw[start:end])), signal_raw[start:end])
        # plt.show()
        # Get BioSPPy ECG object
        try:
            ecg_object = ecg.ecg(signal=signal_raw, sampling_rate=self.fs, show=False)
        except:
            return -1, -1, -1, -1, -1, -1
        # Get BioSPPy output
        ts = ecg_object['ts']          # Signal time array
        rpeaks = ecg_object['rpeaks']  # rpeak indices

        # Get templates and template time array
        templates, rpeaks = self._extract_templates(signal_filtered, rpeaks, template_before, template_after)
        templates_ts = np.linspace(-template_before, template_after, templates.shape[1], endpoint=False)

        # Polarity check
        signal_raw, signal_filtered, templates = self._check_waveform_polarity(polarity_check=polarity_check,
                                                                               signal_raw=signal_raw,
                                                                               signal_filtered=signal_filtered,
                                                                               templates=templates)
        # Normalize waveform
        signal_raw, signal_filtered, templates = self._normalize_waveform_amplitude(normalize=normalize,
                                                                                    signal_raw=signal_raw,
                                                                                    signal_filtered=signal_filtered,
                                                                                    templates=templates)
        return ts, signal_raw, signal_filtered, rpeaks, templates_ts, templates


    def _extract_templates(self, signal_filtered, rpeaks, before, after):

        # convert delimiters to samples
        before = int(before * self.fs)
        after = int(after * self.fs)

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - before
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + after
            if b > length:
                break

            # Append template list
            templates.append(signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    @staticmethod
    def _check_waveform_polarity(polarity_check, signal_raw, signal_filtered, templates):

        """Invert waveform polarity if necessary."""
        if polarity_check:

            # Get extremes of median templates
            templates_min = np.min(np.median(templates, axis=1))
            templates_max = np.max(np.median(templates, axis=1))

            if np.abs(templates_min) > np.abs(templates_max):
                return signal_raw * -1, signal_filtered * -1, templates * -1
            else:
                return signal_raw, signal_filtered, templates
    @staticmethod
    def _normalize_waveform_amplitude(normalize, signal_raw, signal_filtered, templates):
        """Normalize waveform amplitude by the median R-peak amplitude."""
        if normalize:

            # Get median templates max
            templates_max = np.max(np.median(templates, axis=1))

            return signal_raw / templates_max, signal_filtered / templates_max, templates / templates_max
        

    def _apply_filter(self, signal_raw, filter_bandwidth):
        """Apply FIR bandpass filter to waveform."""
        signal_filtered, _, _ = filter_signal(signal=signal_raw, ftype='FIR', band='bandpass',
                                              order=int(0.3 * self.fs), frequency=filter_bandwidth,
                                              sampling_rate=self.fs)
        return signal_filtered
    def _group_features(self,ts, signal_raw, signal_filtered, rpeaks,
                        templates_ts, templates):
        template_before = self.template_before
        template_after = self.template_after

        """Get a dictionary of all ECG features"""

        # Empty features dictionary
        # features = dict()

        # Set ECG file name
        # features['file_name'] = file_name
        features=[]
        # Loop through feature groups
        for feature_group in self.feature_groups:

            # Full waveform features
            if feature_group == 'full_waveform_features':

                # Extract features
                full_waveform_features = FullWaveformFeatures(ts=ts, signal_raw=signal_raw,
                                                              signal_filtered=signal_filtered, rpeaks=rpeaks,
                                                              templates_ts=templates_ts, templates=templates,
                                                              fs=self.fs)
                full_waveform_features.extract_full_waveform_features()

                # Update feature dictionary
                # features.update(full_waveform_features.get_full_waveform_features())
                features.append(full_waveform_features.get_full_waveform_features())

        return features