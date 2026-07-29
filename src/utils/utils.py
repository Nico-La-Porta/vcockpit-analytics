import os
import logging
import json
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.interpolate import PchipInterpolator
import pickle
from scipy.signal import butter, filtfilt
import re
from sklearn.preprocessing import RobustScaler


def extract_session_start_end_timestamp(folder_path):
    """
    Extract the start timestamp of the session from the vr_session.ini file.

    Returns:
    - None
    """
    with open(os.path.join(folder_path, 'vr_session.ini'), 'r') as file:
        for line in file:
            if line.startswith('start_timestamp='):
                session_start_timestamp = int(line.split('=')[1].strip())
            if line.startswith('end_timestamp='):
                session_end_timestamp = int(line.split('=')[1].strip())
                return session_start_timestamp, session_end_timestamp

def import_dataframe(file_name, folder_path=os.getcwd(), sep=None):
    try:
        df = pd.read_csv(os.path.join(folder_path, file_name), sep=sep, engine='python')
        df.columns = [col.strip() for col in df.columns]
    except FileNotFoundError:
        print(f"File '{file_name}' not found in '{folder_path}'")
        df = None
    return df


def interpolate_signal_to_ppg_length(signal, ppg_length):
    """
    Interpolate a signal to match the length of a PPG signal.
    """
    f = PchipInterpolator(np.arange(len(signal)), signal)
    return f(np.linspace(0, len(signal) - 1, ppg_length))

def lowpass_filter(signal, fs):
    """
    Apply a low-pass filter to a signal.
    """
    # signal preprocessing
    ripple, cutoff, stopband, dB_attenuation = 0.1, 5, 10, 30
    fNy = fs / 2  # Nyquist frequency
    Wp = cutoff / fNy
    Ws = stopband / fNy
    N, ft1 = sig.cheb1ord(Wp, Ws, ripple, dB_attenuation)
    b, a = sig.cheby1(N, ripple, ft1, 'low')
    return sig.filtfilt(b, a, signal)


def in_ellipse(x, y, xc, yc, a, b, theta):
    xr = x - xc
    yr = y - yc
    x0 = np.cos(theta) * xr + np.sin(theta) * yr
    y0 = -np.sin(theta) * xr + np.cos(theta) * yr
    p = (x0 ** 2 / a ** 2 + y0 ** 2 / b ** 2) < 1
    return p


def discretize_distraction_signal(signal, window_size, sampling_rate):
    num_windows = len(signal) // window_size

    # Ensure the signal is long enough for at least one window
    if num_windows == 0:
        raise ValueError("Signal is too short for a 1-second window")

    # Divide the signal into 1-second windows and calculate the middle time step for each window
    time_steps = [int((i + 0.5) * window_size / sampling_rate) for i in range(num_windows)]

    # Divide the signal into 1-second windows and calculate the sum for each window
    window_sums = [sum(signal[i * window_size: (i + 1) * window_size]) for i in range(num_windows)]

    return window_sums, time_steps


def reshape_columnwise(arr: np.ndarray, wlen: int = 100, overlap: int = 0) -> np.ndarray:
    if overlap >= wlen:
        raise ValueError("Overlap must be less than the window length")

    step = wlen - overlap
    n_col = (len(arr) - overlap) // step  # Calculate number of columns

    if (len(arr) - overlap) % step != 0:
        n_col += 1  # if there's a remainder, add an additional column

    # Create a new array padded with NaNs to fill the matrix
    padded_length = n_col * step + overlap
    padded_arr = np.full(padded_length, np.nan)
    padded_arr[:len(arr)] = arr

    # Create an array to store the reshaped data
    reshaped_matrix = np.full((wlen, n_col), np.nan)

    for i in range(n_col):
        start_idx = i * step
        end_idx = start_idx + wlen
        reshaped_matrix[:, i] = padded_arr[start_idx:end_idx]

    return reshaped_matrix


def butter_filter(data):
    """
    Applies butter filter to the ppg signal, additionally caps the first 3000 samples to 3*std to compensate initial discrepancies.
    Source:
    """
    total_seconds = data.session_duration_ms /1000
    total_rows = round(len(data.ppg) /10)
    # Calculate the sampling frequency
    fs = total_rows / total_seconds

    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 2.0  # Upper cutoff frequency in Hz
    order = 4  # Filter order
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    filtered_ppg = filtfilt(b, a, data.ppg)

    # capping the first 3000 values for the adjustments of possible initial discrepancies in the signal
    filtered_ppg[:3000] = np.clip(filtered_ppg[:3000], -(filtered_ppg.std()*3), filtered_ppg.std()*3)
    return filtered_ppg


def robust_scaling(data):
    """
    Applies robust scaled to the ppg signal.
    """
    # reshaping as the RobustScaler expects an array of lists with single value
    data_reshaped = data.reshape(-1, 1)

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data_reshaped)
    # reshaping to initial flattened state
    scaled_data = scaled_data.flatten()
    return scaled_data



def extract_value_path(path):
    """
    Extracts value 'n' from 'data_iteration_n.pkl' data
    """
    match = re.search(r'data_iteration_(\d+)\.pkl', path)
    return int(match.group(1)) if match else None


# this is done given the logic that each user did 5 sessions each
# !!!
def info_from_data_iteration(path):
    """
    Given a path, extracts information such as the scenario and user number give the value in the pickle file 'data_iteration_n.pkl'
    (Done using the logic that each user has 5 scenarios each and they are ordered)
    """
    # extracts which scenario it was from filepath
    groups = ["tutorial", "city-slow", "highway-slow", "city-fast", "highway-fast"]
    iteration_number = extract_value_path(path)
    
    if iteration_number is None:
        raise ValueError("Path format is incorrect or iteration number not found.")
    scenario = groups[iteration_number % len(groups)]


    # extracts user number
    iteration_number = extract_value_path(path)
    user_number = iteration_number // 5

    return scenario , user_number +1  # +1 because it starts from zero


def get_mask(data):
    """
    Extracts mask of event / non-event given the data, its ppg values and event-related information
    """
    # mapping event IDs to their start and finish times
    events_dict = {task: [start, end] for task, start, end in zip(data.taskIDs[0], data.event_start_scenario[0], data.event_end_scenario[0])}
    to_delete = ['ReachInitialSpeed', 'TimeoutTask', 'Stop_Call_0', 'Stop_Call_1', 'Stop_Call_2', 'Follow_3',]

    for i in to_delete:
        if i in events_dict:
            del events_dict[i]

    # creating mask
    event_mask = np.zeros(len(data.ppg))
    for key, value in events_dict.items():
        event_mask[value[0] : value[1]] = 1
        
    return event_mask


def perform_feature_extraction(dir_path, window_len_seconds, overlap=0):
    """
    Performs feature extraction on a directory containing pickle files of the session data
    """
    for file in os.listdir(dir_path):
        filename = os.fsdecode(file)
        if filename.endswith(".pkl") and filename.startswith("data_iteration"):
            file_path = os.path.join(dir_path, filename)
            
            # loading data_iteration file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # creating event mask
            event_mask = get_mask(data)

            # getting scenario/user information
            scenario, user =info_from_data_iteration(file_path)
            
            if scenario == 'tutorial':
                continue
            
            #subject = user_info['Subject'].iloc[user-1]
            print(filename, user, scenario)                
            data.extract_features(window_length_seconds=window_len_seconds, overlap=overlap, mask=event_mask, user=user, scenario=scenario, output_file_name=f"{user}_{scenario}.csv")
            print(f"Features extracted for {filename} to {user}_{scenario}.csv")