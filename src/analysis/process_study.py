import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
from collections import defaultdict

# Add the project root to the Python path (for standalone execution)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(os.path.abspath(ROOT_PATH))

from src.app_config.config import *
from src.utils.log_config import logger

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_PATH = PATH['data']
BASE_ROOT = PATH['base_root']

# ----------------------
# --- Util functions ---
# ----------------------

def save_dataframe_pickle(df, filename):
    with open(filename, 'wb') as f:
        pickle.dump(df, f)

def save_dataframe_parquet(df, filename):
    df.to_parquet(filename, engine="pyarrow")

def extract_gazed_objects_names(df, mapping_file='GAZE_TRACK_objects_mapping.json'):
    """
    Adds to dataframe the gazed_objects_names column, 
    a counterpart to the original gazed_objects column with the object names
    """
    def map_gazed_objects(obj_list, mapping):
        if len(obj_list)>0:
            return [mapping.get(int(x), None) if pd.notna(x) else None for x in obj_list]
        return obj_list
    
    with open(os.path.join(BASE_ROOT, f'src/utils/{mapping_file}'), 'r') as file:
        gazed_objects_mapping = json.load(file)
    
    reverse_mapping = {v: k for k, v in gazed_objects_mapping.items()}

    df['gazed_objects_names'] = df['gazed_objects'].apply(lambda x: map_gazed_objects(x, reverse_mapping))

def create_event_and_segment_masks(df):
    """
    Creates event and segment masks for each row in the dataframe.
    - Event mask: boolean array where True indicates the presence of an event.
    - Segment mask: integer array where each segment (event or non-event) is assigned a unique integer ID (1-7).
    """
    event_masks = []
    segment_masks = []
    for _, row in df.iterrows():
        length = len(row['vehicle_straight_driving_offset'])
        
        # Create event mask
        event_mask = np.zeros(length, dtype=bool)
        for start, end in zip(row['event_start_scenario'], row['event_end_scenario']):
            event_mask[start:end] = True
        event_masks.append(event_mask)
        
        # Create segment mask
        segment_mask = np.zeros(length, dtype=int)
        current_segment = 1
        prev_value = False  # Assume session starts with non-event
        
        for i, val in enumerate(event_mask):
            if val != prev_value:
                current_segment += 1
                prev_value = val
            segment_mask[i] = current_segment
        
        segment_masks.append(segment_mask)

    df['event_mask'] = event_masks
    df['segment_mask'] = segment_masks

def count_lane_changes_per_segment_fixed(lane, segment_mask, num_segments=4):
    """
    Count lane changes for each segment, returning fixed-length lists.
    If a segment has no lane changes, 0 is added.
    Caps:
    - Event segments: max 3
    - Non-event segments: max 4
    
    Returns:
    - changes_inside_segments: lane changes for event segments
    - changes_outside_segments: lane changes for non-event segments
    """
    # Dictionary to accumulate lane changes per segment
    segment_changes = defaultdict(int)

    for i in range(1, len(lane)):
        if lane[i] != lane[i-1]:  # lane change detected
            segment_changes[segment_mask[i]] += 1

    # Prepare fixed-length lists
    changes_inside_segments = []
    changes_outside_segments = []

    # Even segments = inside events, odd segments = outside events
    for seg_id in range(1, 2*num_segments, 2):  # odd segments
        changes_outside_segments.append(segment_changes.get(seg_id, 0))
    for seg_id in range(2, 2*num_segments+1, 2):  # even segments
        changes_inside_segments.append(segment_changes.get(seg_id, 0))

    # Cap lengths
    changes_inside_segments = changes_inside_segments[:3]   # max 3 event segments
    changes_outside_segments = changes_outside_segments[:4] # max 4 non-event segments

    return np.array(changes_inside_segments), np.array(changes_outside_segments)

def calculate_violation_percentage(safety_distance_mask, event_mask):
    """
    Calculates the percentage of time the safety distance is kept.
    """
    if len(safety_distance_mask) == 0:
        return None, None

    event_mask = np.array(event_mask)
    inside_event_values = np.array(safety_distance_mask)[event_mask]
    outside_event_values = np.array(safety_distance_mask)[~event_mask]

    inside_event_inside_perc = np.sum(inside_event_values == 1) / len(inside_event_values) * 100
    inside_event_outside_perc = np.sum(inside_event_values == 0) / len(inside_event_values) * 100

    outside_event_inside_perc = np.sum(outside_event_values == 1) / len(outside_event_values) * 100
    outside_event_outside_perc = np.sum(outside_event_values == 0) / len(outside_event_values) * 100

    # returns just the percentage of kept safety distance for both inside and outside the event
    return inside_event_inside_perc, outside_event_inside_perc 

def count_distance_violation_changes_per_segment_fixed(violation_mask, segment_mask, num_segments=4):
    """
    Counts the number of times the safety distance violation status changes 
    from 1 to 0 within each segment.
    """
    
    # Dictionary to accumulate violation_mask changes per segment
    segment_changes = defaultdict(int)

    for i in range(1, len(violation_mask)):
        if violation_mask[i]==0 and violation_mask[i-1]==1:
            segment_changes[segment_mask[i]] += 1
    
    if len(violation_mask)>0:
        if violation_mask[0]==0:
            segment_changes[1] +=1

    # Prepare fixed-length lists
    changes_inside_segments = []
    changes_outside_segments = []

    # Even segments = inside events, odd segments = outside events
    for seg_id in range(1, 2*num_segments, 2):  # odd segments
        changes_outside_segments.append(segment_changes.get(seg_id, 0))
    for seg_id in range(2, 2*num_segments+1, 2):  # even segments
        changes_inside_segments.append(segment_changes.get(seg_id, 0))

    # Cap lengths
    changes_inside_segments = changes_inside_segments[:3]   # max 3 event segments
    changes_outside_segments = changes_outside_segments[:4] # max 4 non-event segments

    return np.array(changes_inside_segments), np.array(changes_outside_segments)



# ----------------------------
# --- Data Fixes functions ---
# ----------------------------

def filter_tasks(acceptable_tasks, tasks, starts, ends, start_subject, scenario, execution, reaction):
    """
    From the original list of tasks, keeps only those in acceptable_tasks.
    As we ignore tasks such as ReachInitialSpeed or Stop_Call.
    """
    # Skip filtering if it's Tutorial
    if scenario == "Tutorial":
        return tasks, starts, ends, execution, reaction

    filtered_tasks = []
    filtered_starts = []
    filtered_ends = []
    filtered_start_subjects = []
    filtered_execution_time = []
    filtered_reaction_time = []

    # Handle nested lists (in case of [[...]])
    if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], (list, np.ndarray)):
        tasks = tasks[0]
    if isinstance(starts, list) and len(starts) > 0 and isinstance(starts[0], (list, np.ndarray)):
        starts = starts[0]
    if isinstance(ends, list) and len(ends) > 0 and isinstance(ends[0], (list, np.ndarray)):
        ends = ends[0]
    if isinstance(start_subject, list) and len(start_subject) > 0 and isinstance(start_subject[0], (list, np.ndarray)):
        start_subject = start_subject[0]
    if isinstance(execution, list) and len(execution) > 0 and isinstance(execution[0], (list, np.ndarray)):
        execution = execution[0]
    if isinstance(reaction, list) and len(reaction) > 0 and isinstance(reaction[0], (list, np.ndarray)):
        reaction = reaction[0]

    for t, s, e, ss, ex, r in zip(tasks, starts, ends, start_subject, execution, reaction):
        if t in acceptable_tasks:  # keep only acceptable tasks
            filtered_tasks.append(t)
            filtered_starts.append(s)
            filtered_ends.append(e)
            filtered_start_subjects.append(ss)
            filtered_execution_time.append(ex)
            filtered_reaction_time.append(r)

    return filtered_tasks, filtered_starts, filtered_ends, filtered_start_subjects, filtered_execution_time, filtered_reaction_time

def upsample_awareness_safety(df, to_print=False):
    """
    As Awareness and Safety are not consistent with the length of the other signals,
    we upsample them to arrive at 10% of the length of other signals (e.g., ppg).
    (This is done because both metrics, )
    """
    for idx, row in df.iterrows():
        ppg_len = len(row['ppg'])

        # Skip NaN or missing data
        if row['awareness'] is None or row['safety'] is None or len(row['awareness'])==0 or len(row['safety'])==0:
            if to_print:
                logger.info(f"User {row['user_id']} {row['scenario']} | is NaN")
            continue

        # Target length
        target_len = int(round(ppg_len / 100))
        og_len = len(row['awareness'])

        # Function to resample using interpolation
        def resample_signal(signal, new_len):
            old_indices = np.linspace(0, 1, len(signal))
            new_indices = np.linspace(0, 1, new_len)
            return np.interp(new_indices, old_indices, signal)

        # Resample both awareness and safety
        awareness_resampled = resample_signal(row['awareness'], target_len)
        safety_resampled = resample_signal(row['safety'], target_len)

        # Store back into dataframe
        df.at[idx, 'awareness'] = awareness_resampled
        df.at[idx, 'safety'] = safety_resampled
        if to_print:
            logger.info(f"User {row['user_id']} {row['scenario']} | {len(row['ppg'])} -> Old: {og_len} - New:{len(safety_resampled)}")

def exclude_segment7(row, cols_to_trim:list, to_print=False):
    """
    Excludes segment 7, trimming all needed columns.
    """
    seg_mask = np.array(row['segment_mask'])
    total_before = len(seg_mask)
    keep_idx = seg_mask != 7     
    total_after = np.sum(keep_idx)
    removed = total_before - total_after

    for col in cols_to_trim:
        if col in row and isinstance(row[col], (list, np.ndarray)):
            arr = np.array(row[col])
            if arr.shape[0] == total_before:
                row[col] = arr[keep_idx].tolist()
    
    # handling awareness and safety 
    if row['awareness'] is not None and row['safety'] is not None and len(row['awareness'])>0 and len(row['safety'])>0:   
        safety_row =np.array(row['safety'])
        awareness_row =np.array(row['awareness'])
        safety_row = safety_row[:-(round(removed/100))]
        awareness_row = awareness_row[:-(round(removed/100))]
        row['awareness'] = awareness_row.tolist()
        row['safety'] = safety_row.tolist()

    if to_print:
        logger.info(f"User {row['user_id']} {row['scenario']} | removed {removed} samples")
    return row

def trim_idle_start(row, cols_to_trim, event_cols):
    """
    Cuts the initial part of each session, where the user is not moving,
    as well as when the vehicle is not entirely in the lane.
    """
    # Find where entering lane
    trim_points = np.where(np.array(row['vehicle_straight_driving_offset'])<=1)[0]
    if len(trim_points) == 0:
        return row
    start_idx = trim_points[0]
    og_len = len(row['ppg'])
    
    # Trim all list/array-like columns that match length
    for col in cols_to_trim:
        if col in row and isinstance(row[col], (list, np.ndarray)):
            arr = np.array(row[col])
            if arr.shape[0] >= start_idx:
                row[col] = arr[start_idx:].tolist()
    
    # Adjust event indices
    for col in event_cols:
        if col in row and isinstance(row[col], list):
            row[col] = [max(0, e - start_idx) for e in row[col]]

    # handling awareness and safety 
    if row['awareness'] is not None and row['safety'] is not None and len(row['awareness'])>0 and len(row['safety'])>0:   
        safety_row =np.array(row['safety'])
        awareness_row =np.array(row['awareness'])
        safety_row = safety_row[round(start_idx/100):]
        awareness_row = awareness_row[round(start_idx/100):]
        row['awareness'] = awareness_row.tolist()
        row['safety'] = safety_row.tolist()
    
    return row

def trim_idle_end(row, cols_to_trim, ending_time):
    """
    Cuts the ending part of each session, where the user has finished
    the tasks and the car stops abruptly.
    """
    original_len = len(row['ppg'])

    # Trim all list/array-like columns that match length
    for col in cols_to_trim:
        if col in row and isinstance(row[col], (list, np.ndarray)):
            arr = np.array(row[col])
            if arr.shape[0] > ending_time:
                row[col] = arr[:ending_time].tolist()
    
    # handling awareness and safety
    if row['awareness'] is not None and row['safety'] is not None and len(row['awareness'])>0 and len(row['safety'])>0:   
        safety_row =np.array(row['safety'])
        awareness_row =np.array(row['awareness'])
        reference_len = round((original_len - ending_time)/100)
        safety_row = safety_row[:-(reference_len)]
        awareness_row = awareness_row[:-(reference_len)]
        row['awareness'] = awareness_row.tolist()
        row['safety'] = safety_row.tolist()

    return row

# -----------------------------
# --- Distraction functions ---
# -----------------------------
    
def compute_fixation_durations(sequence):
    durations = np.zeros_like(sequence, dtype=float)
    current = 0

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current += 1
        else:
            current = 0
        durations[i] = current

    return durations

def compute_tau_adjustment(age, exp, k=0.10):
    """
    Returns a proportional adjustment factor.
    total_score = a + e
    tau_new = tau_base * (1 + adjustment)
    """
    def age_score(age):
        if age > 65:
            return -2
        elif 55 < age <= 65:
            return -1.5
        elif 45 < age <= 55:
            return -1
        elif 35 < age <= 45:
            return 0
        elif 25 < age <= 35:
            return 1
        elif 15 <= age <= 18:
            return 2
        return 0
    
    def experience_score(exp_str):
        """
        exp_str is one of:
        '2-5 years', '5-7 years', '7-10 years', 'more than 10 years'
        """
        if exp_str == "2-5 years":
            return -1
        elif exp_str == "5-7 years":
            return 0
        elif exp_str == "7-10 years":
            return 0
        elif exp_str == "more than 10 years":
            return 1
        return 0

    a = age_score(age)
    e = experience_score(exp)
    total = a + e

    # proportional adjustment
    adjustment = k * total   # can be negative or positive

    return adjustment  # do NOT clamp unless desired

def estimate_tau(row, percentile=75, fs=1000):
    """
    Estimate a good tau (time constant) from the empirical fixation durations.
    Uses the fixation duration 75th percentile (default).
    """
    gazed_objects = np.array(row['gazed_objects'])

    # Compute fixation durations in samples
    durations_samples = compute_fixation_durations(gazed_objects)
    durations_sec = durations_samples / fs

    if len(durations_sec) == 0:
        return 0.8  # fallback

    # Pick a high percentile so distraction saturates around typical long fixations
    tau_est = np.percentile(durations_sec, percentile)

    return tau_est

def compute_tau_for_row(row, percentile=75, fs=1000):
    # Base tau from fixation durations in df row
    base_tau = estimate_tau(row, percentile=percentile, fs=fs)

    # Adjustment percentage
    adj = compute_tau_adjustment(row['age'], row['driving_exp'])

    # Apply weighting
    return round(base_tau * (1 + adj), 2)

def extract_distraction(row, object_weights, smooth_win=51, percentile=75, fs=1000):
    # Convert lists to numpy
    gazed_objects = np.array(row['gazed_objects'])

    # 1. Fixation duration in samples → seconds
    durations_samples = compute_fixation_durations(gazed_objects)
    durations_sec = durations_samples / fs


    # 2. Retrieve object weight per frame
    weights = np.array([
        object_weights.get(int(obj), 0.0) if not np.isnan(obj) else 0.0
        for obj in gazed_objects
    ])

    # 3a. Exponential time constant
    tau = compute_tau_for_row(row, percentile=percentile, fs=fs)

    # Base time factor
    time_factor = np.exp(weights * (durations_sec / tau)) - 1

    # 3b. Invert sign of time_factor if gazed_object index in  1, 3, 9 o 13
    negative_indices = {1, 3, 9, 13}
    for i, obj in enumerate(gazed_objects):
        if not np.isnan(obj) and int(obj) in negative_indices:
            time_factor[i] = -time_factor[i]

    # 4. Main distraction formula
    if len(time_factor) == 0:
        distraction = np.full(time_factor.shape, None, dtype=object)
    else:
        distraction = np.zeros_like(time_factor, dtype=float)
        reference_value = 0
        distraction[0] = time_factor[0]

        for i in range(1, len(time_factor)):
            if gazed_objects[i] != gazed_objects[i-1]:
                # Gaze changed → reset reference value
                reference_value = distraction[i-1]
                distraction_value = reference_value + time_factor[i]
            else:
                # Accumulate from current reference
                distraction_value = reference_value + time_factor[i]

            # capping distraction value
            if distraction_value > 100:
                distraction_value = 100
            elif distraction_value < 0:
                distraction_value = 0
            distraction[i] = distraction_value

        if smooth_win is not None and smooth_win > 1:
            window = np.ones(smooth_win) / float(smooth_win)
            distraction = np.convolve(distraction, window, mode='same')

        # Ensure numerical bounds
        distraction = np.clip(distraction, 0.0, 100.0)

    return np.array(distraction)


# ------------
# --- MAIN ---
# ------------

def main_process_study(cut_segment7=True, cut_starting_section=True, update_distraction=True):
    # --- Loading Study ---
    STUDY_PICKLE_FILE_PATH = PATH['study_pickle_file_path']
    pickle_file = os.path.join(DATA_PATH, STUDY_PICKLE_FILE_PATH)
    with open(pickle_file, 'rb') as f:
        study = pickle.load(f)
    data = study.data

    # --- Converting to pandas dataframe ---
    df = pd.DataFrame()
    df['taskIDs'] = data['taskIDs']
    for key in data.keys():
        if key in ['time_ms', 'time_hr', 'folder_path', 'AwarenessSafety', 'eda', 'temps_list', 'traffic_light_crossed_red', 'traffic_light_time', 'traffic_light_velocity', 'device']:
            continue
        df[key] = data[key]
    
    #----------------------
    # --- Initial fixes ---
    #----------------------
    # Encoding user-related data
    encoding_dr_exp = {'2-5 years': 1, '5-7 years': 2, '7-10 years': 3, 'more than 10 years': 4}
    encoding_dr_freq = {'Every Day': 4, '2/3 times per week': 3, 'Once a week': 2, 'Once every two weeks': 1}

    if df['driving_exp'].apply(lambda x: isinstance(x, str)).any():
        df['driving_exp'] = df['driving_exp'].map(encoding_dr_exp)

    if df['driving_freq'].apply(lambda x: isinstance(x, str)).any():
        df['driving_freq'] = df['driving_freq'].map(encoding_dr_freq)

    # Fixing scenario naming bug
    df["scenario"] = df["scenario"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df["scenario"] = df["scenario"].replace({"City - calls short": "City - calls - short"})

    # adding gazed object names separately
    extract_gazed_objects_names(df)

    # --- Removing Tutorial sessions and ordering dataframe ---
    scenarios = ['Tutorial', 'City - calls', 'Highway - music', 'City - calls - short', 'Highway - music - short']
    df["scenario"] = pd.Categorical(df["scenario"], categories=scenarios, ordered=True)
    df = df.sort_values(by=["user_id", "scenario"]).reset_index(drop=True)
    df = df.reindex(columns=["user_id", "scenario"] +
                         [c for c in df.columns if c not in ["user_id", "scenario"]])

    df = df[df['scenario'] != 'Tutorial']

    # --- Selecting only relevant tasks of sessions (Call / Play) ---
    ending_session_time = [row['event_start_scenario'][-1] for idx, row in df.iterrows()] # for cutting ending idle section if segment 7 is not cut
    acceptable_tasks = ['Call_0', 'Call_1', 'Call_2', 'Play_0', 'Play_1', 'Play_2']
    df[['taskIDs', 'event_start_scenario', 'event_end_scenario', 'event_start_subject', 'event_execution_time', 'event_reaction_time']] = df.apply(
        lambda row: pd.Series(filter_tasks(acceptable_tasks, row['taskIDs'], row['event_start_scenario'], row['event_end_scenario'], row['event_start_subject'], row['scenario'], row['event_execution_time'], row['event_reaction_time'])),axis=1)
    
    # --- Creating masks (Event & Segment) ---
    create_event_and_segment_masks(df)

    # Fixing 'lane' column
    df['lane'] = df['lane'].apply(lambda x: [int(i) for i in x])

    # Adding column with timestamps in seconds
    df['time_sec'] = df.apply(lambda row: np.linspace(0, (row['session_end_timestamp'] - row['session_start_timestamp']) / 1000, len(row['vehicle_speed'])), axis=1)

    # Replacing NaNs in gazed_objects with 15, being 'Other'
    for idx, row in df.iterrows():
        arr_row = np.array(row['gazed_objects'], dtype=float)
        arr_row = np.nan_to_num(arr_row, nan=15)  # Replace NaN with 15
        df.at[idx, 'gazed_objects'] = arr_row.tolist()

    df["folder_paths"] = df["folder_paths"].astype(str)
    
    logger.info(">>> Initial fixes applied successfully.")

    # --------------------------
    # --- Session processing ---
    # --------------------------

    # --- Upsampling Awareness and Safety to be the same length as
    # other columns but divided by 100 ---
    upsample_awareness_safety(df, to_print=False)
    logger.info(">>> Awareness and Safety upsampled.")

    # --- Removing the 7th segment from sessions if condition ---
    cols_to_trim = ['involvement', 'ppg', 'gaze_through_array', 'category', 'gazed_objects', 'gazed_objects_names', 
                    'distraction','vehicle_speed', 'vehicle_safety_distance_mask', 'distance',
                    'vehicle_straight_driving_offset', 'lane', 'focus', 'event_mask', 'segment_mask', 'time_sec']
    
    if cut_segment7:
        df = df.apply(lambda row: exclude_segment7(row, cols_to_trim, to_print=False), axis=1)
        logger.info(">>> Segment 7 removed")
    else:
        for i, (idx, row) in enumerate(df.iterrows()):
            df.loc[idx] = trim_idle_end(row, cols_to_trim, ending_session_time[i])
        logger.info(">>> Idle Ending removed")

    # --- Removing the starting section ---
    event_cols = ['event_start_scenario', 'event_end_scenario', 'event_start_subject']
    if cut_starting_section:
        df = df.apply(trim_idle_start, axis=1, args=(cols_to_trim, event_cols))
        logger.info(">>> Starting section cut")

    # --- N. Segments ---
    # We set a number of segments for calculating both lane changes and safety violations based on the removal or not of segment 7
    if cut_segment7:
        num_segments = 3
    else:
        num_segments = 4

    # --- Lane Changes ---
    lane_df = df[df['scenario'].isin(['Highway - music', 'Highway - music - short'])].copy(deep=True)
    lane_df[['changes_inside_event', 'changes_outside_event']] = lane_df.apply(lambda row: pd.Series(count_lane_changes_per_segment_fixed(row['lane'], row['segment_mask'], num_segments)), axis=1)
    lane_df = lane_df.sort_values(['user_id', 'scenario'])
    df = pd.merge(df, lane_df[['user_id', 'scenario', 'changes_inside_event', 'changes_outside_event']], on=['user_id', 'scenario'], how='left')
    df = df.rename(columns={'changes_inside_event': 'lane_changes_event', 'changes_outside_event': 'lane_changes_outside_event'})
    logger.info(">>> Lane Changes calculated")

    # --- Safety Distance violations ---
    df[['safety_distance_violations_event', 'safety_distance_violations_outside_event']] = df.apply(
        lambda row: pd.Series(count_distance_violation_changes_per_segment_fixed(row['vehicle_safety_distance_mask'], row['segment_mask'], num_segments)),axis=1)

    df[['safety_distance_kept_perc_event', 'safety_distance_kept_perc_outside_event']] = df.apply(
        lambda row: pd.Series(calculate_violation_percentage(row['vehicle_safety_distance_mask'], row['event_mask'])),axis=1)
    logger.info(">>> Safety Distance violations calculated")

    # --- gaze_category_mask creation ---
    # creating a mask for when the user is looking at the windscreen or not 
    # (later used for percentages)
    gaze_category_masks = []
    for _, row in df.iterrows():
        category_mask = np.zeros(len(row['category']), dtype=bool)
        for idx, value in enumerate(row['category']):
            if value == 1:
                category_mask[idx] = True
            else:
                category_mask[idx] = False
        gaze_category_masks.append(category_mask)

    df['gaze_category_mask'] = gaze_category_masks

    # --------------------------
    # --- Update Distraction ---
    # --------------------------
    fs=1000
    object_weights = {
        1: 0.5,     # WINDSCREEN        #POSITIVE
        2: 1.1,     # BUILDING
        3: 1.3,     # CAR               #POSITIVE
        4: 1.5,     # COCKPIT
        5: 1.3,     # DOOR
        6: 0.2,     # LEFT_MIRROR
        7: 0.2,     # REARVIEW MIRROR
        8: 0.2,     # RIGHT_MIRROR
        9: 1.5,     # ROAD              #POSITIVE
        10: 2.5,    # SCREEN_SCREEN_000
        11: 2.5,    # SCREEN_SCREEN_001
        12: 1.0,    # SKYBOX
        13: 0.7,    # STEERING_WHEEL    # POSITIVE
        14: 1.5}    # TABLET

    if update_distraction:
        df['distraction'] = df.apply(lambda r: extract_distraction(r, object_weights, percentile=50, fs=fs), axis=1)
        logger.info(">>> Distraction updated")
    
    # ------------------------
    # --- Saving dataframe ---
    # ------------------------
    data_pickle_file = os.path.join(DATA_PATH, 'processed_dataframe.pkl')
    data_parquet_file = os.path.join(DATA_PATH, 'processed_dataframe.parquet')

    save_dataframe_pickle(df, data_pickle_file)
    save_dataframe_parquet(df, data_parquet_file)
    logger.info(f">>> Processed dataframe saved to {data_pickle_file} & {data_parquet_file}")

if __name__ == "__main__":
    cut_segment7 = False
    cut_starting_section = True
    update_distraction = True
    main_process_study(cut_segment7=cut_segment7, cut_starting_section=cut_starting_section, update_distraction=update_distraction)