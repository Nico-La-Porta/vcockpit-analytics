import os

import numpy as np
import pandas as pd
import pyhrv
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
import pyhrv.time_domain as td
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import tsfel
import warnings


def extract_all_features(data, output_file_name="features.csv", output_dir=os.getcwd(), mask=None, user=None, scenario=None):
    """
    Extract all features from a DataFrame column

    Parameters
    ----------
    data : np.ndarray
        Matrix containing the windowed PPG signal (n_window, window_length)
    """

    # Extract statistical features
    stats_features = get_statistic_features(data)
    stats_features.reset_index(drop=True, inplace=True)
    # Extract time-domain features
    time_domain_features = get_time_domain_features(data)
    time_domain_features.reset_index(drop=True, inplace=True)
    # Extract frequency-domain features
    frequency_domain_features = get_frequency_domain_features(data)
    frequency_domain_features.reset_index(drop=True, inplace=True)
    # Extract non-linear features
    non_linear_features = get_non_linear_features(data)
    non_linear_features.reset_index(drop=True, inplace=True)

    temporal_features = get_temporal_features(data)
    spectral_features = get_spectral_features(data)
    fractal_features = get_fractal_features(data)

    # Concatenate all features into a single DataFrame
    final_df = pd.concat([stats_features, time_domain_features, frequency_domain_features, non_linear_features,
                          temporal_features, spectral_features, fractal_features], axis=1) # , fractal_features
    
    final_df.dropna(inplace=True)

    

    print(f'DF LEN: {len(final_df)}')
    #adding optional information
    if mask is not None:
        final_df['label'] = mask

    if user is not None:
        final_df['user_id'] = user

    if scenario is not None:
        final_df['scenario'] = scenario

    # Save the final DataFrame to a pickle file     
    final_df.to_csv(os.path.join(output_dir, output_file_name), index=False) 


def zero_crossings(data):
    """ Find the indices of zero crossings in a signal """
    # np.diff computes the difference between consecutive elements
    # np.sign returns an element-wise indication for the sign of each element
    # np.where finds the indices of non-zero elements
    zero_crossings_indices = np.where(np.diff(np.sign(data)))[0]
    return zero_crossings_indices


def find_my_peaks(data):
    """ Find peaks in a signal using zero-crossing rate as a threshold """
    crossings = zero_crossings(data)
    zero_crossing_rate = np.diff(crossings)
    # find_peaks finds peaks in a signal based on distance and prominence criteria
    peaks, _ = find_peaks(data, distance=np.mean(zero_crossing_rate), prominence=0.5)
    return peaks


def tsfel_framework(data: np.ndarray, feature_type: str):
    cfg = tsfel.get_features_by_domain(feature_type)
    tsfel_features = pd.DataFrame()

    for i in range(data.shape[1]):
        subset = pd.Series(data[:, i])
        
        # Replace NaN values with the median
        if subset.isna().any():
            subset = subset.fillna(subset.median())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                # Compute tsfel statistical features for each subset
                tsfel_subset = tsfel.time_series_features_extractor(cfg, subset, verbose=0)
                tsfel_features = pd.concat([tsfel_features, tsfel_subset], ignore_index=True)
            except TypeError as e:
                print(f"Skipped subset {i} due to error: {e}")
                continue

    # Remove unwanted prefix in column names
    tsfel_features.columns = tsfel_features.columns.str.replace('0_', '')
    return tsfel_features


def get_time_domain_features(data):
    """ Compute time-domain features for IBI signals"""
    # Initialize a DataFrame with specific columns
    df_features = pd.DataFrame(columns=["nni_counter", "nni_mean", "nni_min", "nni_max", "nni_diff_mean",
                                        "nni_diff_min", "nni_diff_max", "hr_mean", "hr_min", "hr_max", "hr_std",
                                        "sdnn", "rmssd", "sdsd", "tinn_n", "tinn_m", "tinn", "tri_index", "nn20",
                                        "pnn20", "nn40", "pnn40"])

    for ind in range(data.shape[1]):
        ppg_signal = data[:, ind]
        peaks = find_my_peaks(ppg_signal)
        # Initialize features to zero if no peaks are found
        features = [0] * len(df_features.columns)
        if len(peaks) > 1:
            t = np.arange(len(ppg_signal))
            nni = pyhrv.tools.nn_intervals(t[peaks])
            if len(nni) > 2:
                features = []
                # Append various HRV time-domain features
                features.extend([v for v in td.nni_parameters(nni)])
                features.extend([v for v in td.nni_differences_parameters(nni)])
                features.extend([v for v in td.hr_parameters(nni)])
                features.extend([v for v in td.sdnn(nni)])
                features.extend([v for v in td.rmssd(nni)])
                features.extend([v for v in td.sdsd(nni)])
                features.extend([v for v in td.geometrical_parameters(nni, plot=False)[1:]])
                features.extend([v for v in td.nn20(nni)])
                features.extend([v for v in td.nnXX(nni, threshold=40)])
        # Concatenate the features to the DataFrame
        df_features = pd.concat([df_features, pd.DataFrame([features], columns=df_features.columns)])
    return df_features


def calculate_fd_results(nni):
    """ Calculate frequency-domain results from NN intervals """
    f_results = []
    results = fd.lomb_psd(nni, show=False)
    plt.close('all')  # Close the plot generated by lomb_psd
    for v in results[1:6]:
        f_results.extend(list(v))
    f_results += list(results[6:8])
    return f_results


def get_frequency_domain_features(data):
    """ Compute frequency-domain features for heart rate variability """
    # Initialize a DataFrame with specific columns
    df_features = pd.DataFrame(columns=["lomb_peak_ulf", "lomb_peak_lf", "lomb_peak_hf", "lomb_abs_ulf", "lomb_abs_lf",
                                        "lomb_abs_hf", "lomb_rel_ulf", "lomb_rel_lf", "lomb_rel_hf", "lomb_log_ulf",
                                        "lomb_log_lf", "lomb_log_hf", "lomb_norm1", "lomb_norm2", "lomb_ratio",
                                        "lomb_total"])

    for ind, signal in enumerate(data):
        peaks = find_my_peaks(signal)
        # Initialize features to zero if no peaks are found
        features = [0] * len(df_features.columns)
        if len(peaks) > 1:
            t = np.arange(len(signal))
            nni = pyhrv.tools.nn_intervals(t[peaks])
            if len(nni) > 1:
                features = calculate_fd_results(nni)
        # Concatenate the features to the DataFrame
        df_features = pd.concat([df_features, pd.DataFrame([features], columns=df_features.columns)])
    return df_features


def get_non_linear_features(data):
    """ Compute non-linear features for heart rate variability"""
    # Initialize a DataFrame with specific columns
    df_features = pd.DataFrame(columns=["sd1", "sd2", "sd_ratio", "ellipse_area"])

    for ind, signal in enumerate(data):
        peaks = find_my_peaks(signal)
        # Initialize features to zero if no peaks are found
        features = [0] * len(df_features.columns)
        if len(peaks) > 1:
            t = np.arange(len(signal))
            nni = pyhrv.tools.nn_intervals(t[peaks])
            if len(nni) > 1:
                features = [v for v in nl.poincare(nni, show=False)[1:]]
                plt.close('all')  # Close the plot generated by poincare
        # Concatenate the features to the DataFrame
        df_features = pd.concat([df_features, pd.DataFrame([features], columns=df_features.columns)])
    return df_features


def get_statistic_features(data: np.ndarray) -> pd.DataFrame:
    '''Computes all statistical features of PPG signals using tsfel library.'''
    feature_type = 'statistical'
    stats_features = tsfel_framework(data, feature_type)
    return stats_features


def get_temporal_features(data: np.ndarray) -> pd.DataFrame:
    '''Computes all temporal features of PPG signals using tsfel library.'''
    feature_type = 'temporal'
    temporal_features = tsfel_framework(data, feature_type)
    return temporal_features


def get_spectral_features(data: np.ndarray) -> pd.DataFrame:
    '''Computes all spectral features of PPG signals using tsfel library.'''
    feature_type = 'spectral'
    spectral_features = tsfel_framework(data, feature_type)
    return spectral_features


def get_fractal_features(data: np.ndarray) -> pd.DataFrame:
    '''Computes all fractal features of PPG signals using tsfel library.'''
    feature_type = 'fractal'
    fractal_features = tsfel_framework(data, feature_type)
    return fractal_features