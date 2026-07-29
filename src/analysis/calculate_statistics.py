import pandas as pd
import numpy as np
import os
import pickle
import sys
import json

# Add the project root to the Python path (for standalone execution)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(os.path.abspath(ROOT_PATH))

from src.app_config.config import *
from src.utils.log_config import logger
from src.analysis.process_study import save_dataframe_pickle

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


DATA_PATH = PATH['data']
BASE_ROOT = PATH['base_root']

# ----------------------
# --- Util functions ---
# ----------------------

def load_dataframe(file_path:str, file_type:str='pickle'):
    """
    Loads pickle/parquet file containing output Dataframe from "process_study.py".
    """
    if file_type == 'pickle':
        with open(file_path, 'rb') as f:
            dataframe = pickle.load(f)
    elif file_type == 'parquet':
        dataframe = pd.read_parquet(file_path)
    return dataframe


# -------------------------
# --- Stats computation ---
# -------------------------

def compute_segment_slope(values, time_values):
    if len(values) < 2:
        return np.nan

    # Remove NaNs
    mask = ~np.isnan(values) & ~np.isnan(time_values)
    values = values[mask]
    time_values = time_values[mask]

    if len(values) < 2:
        return np.nan

    return (values[-1] - values[0]) / (time_values[-1] - time_values[0])

def compute_segment_stats_per_session(df, top_n_gazed=3, cut_segment7=False):
    """
    Compute mean/std for each segment (1–6/7) and overall Event / Non-Event
    per session (per row), including offset variance, awareness/safety stats,
    and top 3 gazed objects per segment and overall.
    """
    cols = ['vehicle_speed', 'focus', 'distraction', 'involvement'] # cols for mean/std/quantiles
    if cut_segment7:
        segment_labels = {1: 'NE1', 2: 'E1', 3: 'NE2', 4: 'E2', 5: 'NE3', 6: 'E3'}
    else:
        segment_labels = {1: 'NE1', 2: 'E1', 3: 'NE2', 4: 'E2', 5: 'NE3', 6: 'E3', 7: 'NE4'}
    all_stats = []

    # Columns "array-like" for calculating global rate
    array_cols = cols + ['awareness', 'safety', 
                         'vehicle_straight_driving_offset',
                         'gaze_category_mask',
                         'vehicle_safety_distance_mask']
    global_dynamic_ranges = {}
    for col in array_cols:
        all_values = []
        for _, row in df.iterrows():
            raw = row.get(col, [])
            if hasattr(raw, '__len__') and not isinstance(raw, (str, float, int)):
                all_values.extend(np.array(raw, dtype=np.float64).tolist())
        arr = np.array(all_values, dtype=np.float64)
        r = np.nanmax(arr) - np.nanmin(arr)
        global_dynamic_ranges[col] = r if r > 0 else 1  # avoids division by zero

    for row_idx, row in df.iterrows():
        session_stats = {"user_id": row["user_id"], "scenario": row["scenario"]}

        seg_mask_full = np.array(row["segment_mask"])
        event_mask_full = np.array(row["event_mask"], dtype=bool)

        # --- SEGMENT LENGTH (in seconds) ---
        time_sec = np.array(row['time_sec'])
        segments = np.unique(seg_mask_full)

        # Compute duration of each segment
        segment_durations = {}
        for seg in segments:
            mask = seg_mask_full == seg
            segment_times = time_sec[mask]
            duration = segment_times[-1] - segment_times[0]  # last - first
            segment_durations[seg] = round(duration, 2)
        for seg in segment_durations:
            session_stats[f"time_sec_{segment_labels[seg]}"] = segment_durations[seg]
        if len(segment_durations)>0:
            session_stats["time_sec_E_all"] = segment_durations[2] + segment_durations[4] + segment_durations[6]
            if cut_segment7:
                session_stats["time_sec_NE_all"] = segment_durations[1] + segment_durations[3] + segment_durations[5]
            else:
                session_stats["time_sec_NE_all"] = segment_durations[1] + segment_durations[3] + segment_durations[5] + segment_durations[7]
        else:
            session_stats["time_sec_E_all"] = None
            session_stats["time_sec_NE_all"] = None
        # --- MEAN / STD_DEV / QUANTILES FOR MAIN COLS ---
        for col in cols:
            values = row.get(col)
            if values is None or not hasattr(values, '__len__') or len(values) != len(seg_mask_full):
                values = np.full(len(seg_mask_full), np.nan, dtype=np.float64)
            else:
                values = np.array(values, dtype=np.float64)

            for seg_num, seg_label in segment_labels.items():
                seg_values = values[seg_mask_full == seg_num]
                session_stats[f"{col}_{seg_label}_mean"] = np.nanmean(seg_values)
                session_stats[f"{col}_{seg_label}_std"]  = np.nanstd(seg_values)
                if col in ['focus', 'distraction']:
                    session_stats[f"{col}_{seg_label}_q25"] = np.nanquantile(seg_values, 0.25)
                    session_stats[f"{col}_{seg_label}_median"] = np.nanmedian(seg_values)
                    session_stats[f"{col}_{seg_label}_q75"] = np.nanquantile(seg_values, 0.75)

                time_values = time_sec[seg_mask_full == seg_num]

                slope = compute_segment_slope(seg_values, time_values)
                dyn_range = global_dynamic_ranges.get(col, 1)
                session_stats[f"{col}_{seg_label}_rate"] = (slope / dyn_range) * 100


            session_stats[f"{col}_E_all_mean"]  = np.nanmean(values[event_mask_full])
            session_stats[f"{col}_E_all_std"]   = np.nanstd(values[event_mask_full])
            session_stats[f"{col}_NE_all_mean"] = np.nanmean(values[~event_mask_full])
            session_stats[f"{col}_NE_all_std"]  = np.nanstd(values[~event_mask_full])
            if col in ['focus', 'distraction']:
                session_stats[f"{col}_E_all_q25"]  = np.nanquantile(values[event_mask_full], 0.25)
                session_stats[f"{col}_E_all_median"]  = np.nanmedian(values[event_mask_full])
                session_stats[f"{col}_E_all_q75"]  = np.nanquantile(values[event_mask_full], 0.75)
                session_stats[f"{col}_NE_all_q25"] = np.nanquantile(values[~event_mask_full], 0.25)
                session_stats[f"{col}_NE_all_median"] = np.nanmedian(values[~event_mask_full])
                session_stats[f"{col}_NE_all_q75"] = np.nanquantile(values[~event_mask_full], 0.75)

        # --- OFFSET VARIANCE ---
        offset_values = np.array(row.get('vehicle_straight_driving_offset', []), dtype=np.float64)
        if len(offset_values) != len(seg_mask_full):
            offset_values = np.full(len(seg_mask_full), np.nan)

        session_stats['vehicle_offset_var_E_all']  = np.nanvar(offset_values[event_mask_full])
        session_stats['vehicle_offset_var_NE_all'] = np.nanvar(offset_values[~event_mask_full])

        for seg_num, seg_label in segment_labels.items():
            seg_values = offset_values[seg_mask_full == seg_num]
            session_stats[f"vehicle_offset_var_{seg_label}"] = np.nanvar(seg_values)
            time_values = time_sec[seg_mask_full == seg_num]

            slope = compute_segment_slope(seg_values, time_values)
            dyn_range = global_dynamic_ranges.get('vehicle_straight_driving_offset', 1)
            session_stats[f"vehicle_offset_var_{seg_label}_rate"] = (slope / dyn_range) * 100

        # --- SAFETY VIOLATIONS ---
        sdv_e = np.array(row['safety_distance_violations_event'])
        sdv_ne = np.array(row['safety_distance_violations_outside_event'])
        for i, seg_label in enumerate(['E1', 'E2', 'E3']):
            session_stats[f"safety_distance_violations_{seg_label}"] = sdv_e[i]
        if cut_segment7:  
            for i, seg_label in enumerate(['NE1', 'NE2', 'NE3']):
                session_stats[f"safety_distance_violations_{seg_label}"] = sdv_ne[i]
        else:
            for i, seg_label in enumerate(['NE1', 'NE2', 'NE3', 'NE4']):
                session_stats[f"safety_distance_violations_{seg_label}"] = sdv_ne[i]
        session_stats["safety_distance_violations_E_all"]  = sdv_e.sum()
        session_stats["safety_distance_violations_NE_all"] = sdv_ne.sum()
        session_stats['safety_distance_kept_E_all'] = row['safety_distance_kept_perc_event']
        session_stats['safety_distance_kept_NE_all'] = row['safety_distance_kept_perc_outside_event']

        # --- AWARENESS / SAFETY ---
        for col in ['awareness', 'safety']:
            raw_values = row.get(col, [])
            if not hasattr(raw_values, '__len__') or isinstance(raw_values, (float, int, np.floating, np.integer)):
                values = np.array([], dtype=np.float64)
            else:
                values = np.array(raw_values, dtype=np.float64)
            if len(values) == 0:
                continue

            idxs = np.linspace(0, len(seg_mask_full) - 1, len(values)).astype(int)
            seg_mask_ds = seg_mask_full[idxs]
            event_mask_ds = event_mask_full[idxs]

            for seg_num, seg_label in segment_labels.items():
                seg_values = values[seg_mask_ds == seg_num]
                session_stats[f"{col}_{seg_label}_mean"] = np.nanmean(seg_values)
                session_stats[f"{col}_{seg_label}_std"]  = np.nanstd(seg_values)
                session_stats[f"{col}_{seg_label}_q25"]  = np.nanquantile(seg_values, 0.25)
                session_stats[f"{col}_{seg_label}_median"]  = np.nanmedian(seg_values)
                session_stats[f"{col}_{seg_label}_q75"]  = np.nanquantile(seg_values, 0.75)
                time_values = time_sec[idxs][seg_mask_ds == seg_num]
                slope = compute_segment_slope(seg_values, time_values)
                dyn_range = global_dynamic_ranges.get(col, 1)
                session_stats[f"{col}_{seg_label}_rate"] = (slope / dyn_range) * 100
                

            session_stats[f"{col}_E_all_mean"]  = np.nanmean(values[event_mask_ds])
            session_stats[f"{col}_E_all_std"]   = np.nanstd(values[event_mask_ds])
            session_stats[f"{col}_E_all_q25"]   = np.nanquantile(values[event_mask_ds], 0.25)
            session_stats[f"{col}_E_all_median"]   = np.nanmedian(values[event_mask_ds])
            session_stats[f"{col}_E_all_q75"]   = np.nanquantile(values[event_mask_ds], 0.75)
            session_stats[f"{col}_NE_all_mean"] = np.nanmean(values[~event_mask_ds])
            session_stats[f"{col}_NE_all_std"]  = np.nanstd(values[~event_mask_ds])
            session_stats[f"{col}_NE_all_q25"]  = np.nanquantile(values[~event_mask_ds], 0.25)
            session_stats[f"{col}_NE_all_median"]  = np.nanmedian(values[~event_mask_ds])
            session_stats[f"{col}_NE_all_q75"]  = np.nanquantile(values[~event_mask_ds], 0.75)

        # --- TOP 3 GAZED OBJECTS (same frequency as vehicle_speed) ---
        gazed_objects_full = np.array(row.get('gazed_objects_names', []), dtype=object)

        # Per-segment top 3
        for seg_num, seg_label in segment_labels.items():
            seg_objects = gazed_objects_full[seg_mask_full == seg_num]
            # Filter out invalid entries
            seg_objects = np.array([obj for obj in seg_objects if obj is not None and str(obj).lower() != 'nan'])
            if len(seg_objects) > 0:
                unique, counts = np.unique(seg_objects, return_counts=True)
                sorted_idx = np.argsort(-counts)
                top3_objs = unique[sorted_idx[:top_n_gazed]].tolist()
                top3_perc = np.round(counts[sorted_idx[:top_n_gazed]] / len(seg_objects) * 100, 2).tolist()
                session_stats[f'gazed_top{top_n_gazed}_{seg_label}_obj'] = top3_objs
                session_stats[f'gazed_top{top_n_gazed}_{seg_label}_perc'] = top3_perc

        # Event / Non-event top 3
        event_objects = gazed_objects_full[event_mask_full]
        event_objects = np.array([obj for obj in event_objects if obj is not None and str(obj).lower() != 'nan'])
        if len(event_objects) > 0:
            unique, counts = np.unique(event_objects, return_counts=True)
            sorted_idx = np.argsort(-counts)
            session_stats[f'gazed_top{top_n_gazed}_E_obj'] = unique[sorted_idx[:top_n_gazed]].tolist()
            session_stats[f'gazed_top{top_n_gazed}_E_perc'] = np.round(counts[sorted_idx[:top_n_gazed]] / len(event_objects) * 100,2).tolist()

        ne_objects = gazed_objects_full[~event_mask_full]
        ne_objects = np.array([obj for obj in ne_objects if obj is not None and str(obj).lower() != 'nan'])
        if len(ne_objects) > 0:
            unique, counts = np.unique(ne_objects, return_counts=True)
            sorted_idx = np.argsort(-counts)
            session_stats[f'gazed_top{top_n_gazed}_NE_obj'] = unique[sorted_idx[:top_n_gazed]].tolist()
            session_stats[f'gazed_top{top_n_gazed}_NE_perc'] = np.round(counts[sorted_idx[:top_n_gazed]] / len(ne_objects) * 100,2).tolist()

        # --- GAZED CATEGORY ---
        gaze_cat_mask = np.array(row.get('gaze_category_mask', []), dtype=bool)
        if len(gaze_cat_mask) != len(seg_mask_full):
            gaze_cat_mask = np.full(len(seg_mask_full), np.nan, dtype=bool)

        for seg_num, seg_label in segment_labels.items():
            seg_values = gaze_cat_mask[seg_mask_full == seg_num]
            seg_time = time_sec[seg_mask_full == seg_num]
            if len(seg_values) > 0:
                total = len(seg_values)
                perc_windscreen = np.round(seg_values.sum() / total * 100, 2)  # True = windscreen
                perc_else = np.round((total - seg_values.sum()) / total * 100, 2)  # False = else
                if perc_windscreen > 100 or perc_windscreen < 0:
                    print("XXXXXXXXXXXXXXXXXXXX")
                    print(f"user {row["user_id"]},scenario: {row["scenario"]}")
                    print("WINDSCREEN PERC WRONG")
                    print("XXXXXXXXXXXXXXXXXXXX")
                if perc_else > 100 or perc_else < 0:
                    print("XXXXXXXXXXXXXXXXXXXX")
                    print(f"user {row["user_id"]},scenario: {row["scenario"]}")
                    print("PERC ELSE WRONG")
                    print("XXXXXXXXXXXXXXXXXXXX")
                session_stats[f'gaze_category_windscreen_perc_{seg_label}'] = perc_windscreen
                session_stats[f'gaze_category_else_perc_{seg_label}'] = perc_else
                slope = compute_segment_slope(seg_values.astype(float), seg_time)
                dyn_range = global_dynamic_ranges.get('gaze_category_mask', 1)
                session_stats[f"gaze_category_windscreen_perc_{seg_label}_rate"] = (slope / dyn_range) * 100


        # Total Event / Non-Event
        event_values = gaze_cat_mask[event_mask_full]
        ne_values = gaze_cat_mask[~event_mask_full]

        if len(event_values) > 0:
            total = len(event_values)
            session_stats['gaze_category_windscreen_perc_E'] = np.round(event_values.sum() / total * 100, 2)
            session_stats['gaze_category_else_perc_E'] = np.round((total - event_values.sum()) / total * 100, 2)

        if len(ne_values) > 0:
            total = len(ne_values)
            session_stats['gaze_category_windscreen_perc_NE_'] = np.round(ne_values.sum() / total * 100, 2)
            session_stats['gaze_category_else_perc_NE'] = np.round((total - ne_values.sum()) / total * 100, 2)

        # --- SAFETY DISTANCE PERC ---
        safety_mask = np.array(row.get('vehicle_safety_distance_mask', []), dtype=int)
        if len(safety_mask) != len(seg_mask_full):
            safety_mask = np.full(len(seg_mask_full), np.nan, dtype=int)

        for seg_num, seg_label in segment_labels.items():
            seg_values = safety_mask[seg_mask_full == seg_num]
            if len(seg_values) > 0:
                total = len(seg_values)
                perc_within = np.round((seg_values == 1).sum() / total * 100, 2)
                perc_outside = np.round((seg_values == 0).sum() / total * 100, 2)
                session_stats[f'safety_distance_perc_within_{seg_label}'] = perc_within
                session_stats[f'safety_distance_perc_outside_{seg_label}'] = perc_outside
                time_values = time_sec[seg_mask_full == seg_num]
                slope = compute_segment_slope((seg_values == 1).astype(float), time_values)
                dyn_range = global_dynamic_ranges.get('vehicle_safety_distance_mask', 1)
                session_stats[f"safety_distance_perc_within_{seg_label}_rate"] = (slope / dyn_range) * 100

        # Total Event / Non-Event
        event_values = safety_mask[event_mask_full]
        ne_values = safety_mask[~event_mask_full]

        if len(event_values) > 0:
            total = len(event_values)
            session_stats['safety_distance_perc_within_E'] = np.round((event_values == 1).sum() / total * 100, 2)
            session_stats['safety_distance_perc_outside_E'] = np.round((event_values == 0).sum() / total * 100, 2)

        if len(ne_values) > 0:
            total = len(ne_values)
            session_stats['safety_distance_perc_within_NE'] = np.round((ne_values == 1).sum() / total * 100, 2)
            session_stats['safety_distance_perc_outside_NE'] = np.round((ne_values == 0).sum() / total * 100, 2)


        all_stats.append(session_stats)

    return pd.DataFrame(all_stats)

def compute_overall_stats_per_scenario(df_stats):
    """
    Compute overall statistics per scenario from the session-level statistics DataFrame.
    """
    filtered_df_TOTAL = df_stats.copy(deep=True)

    # Columns to exclude from numeric operations
    exclude_cols = ['user_id', 'scenario']

    # Select only numeric columns for averaging
    numeric_cols = filtered_df_TOTAL.select_dtypes(include='number').columns.tolist()

    # Group by scenario and compute mean for numeric columns
    mean_df = filtered_df_TOTAL.groupby('scenario')[numeric_cols].mean().reset_index()

    # Add 'user' column
    mean_df['user_id'] = 'Total'

    # Add any missing columns from the original dataframe with NaN
    for col in filtered_df_TOTAL.columns:
        if col not in mean_df.columns:
            mean_df[col] = pd.NA

    # Reorder columns to match original dataframe
    final_df = mean_df[filtered_df_TOTAL.columns]
    return final_df

def compute_gazed_objects_per_scenario(df, top_n_gazed=3, cut_segment7=False):
    """
    Compute top gazed objects per scenario and segment, including overall E / NE.
    Returns a DataFrame with lists of top objects and percentages.
    """
    if cut_segment7:
        segment_labels = {1: 'NE1', 2: 'E1', 3: 'NE2', 4: 'E2', 5: 'NE3', 6: 'E3'}
    else:
        segment_labels = {1: 'NE1', 2: 'E1', 3: 'NE2', 4: 'E2', 5: 'NE3', 6: 'E3', 7: 'NE4'}
    scenario_stats = []

    for scenario, df_s in df.groupby('scenario'):
        stats = {'scenario': scenario}

        # Collect all gazed objects per segment and overall
        seg_objects_all = {seg: [] for seg in segment_labels.values()}
        E_objects_all = []
        NE_objects_all = []

        for _, row in df_s.iterrows():
            gazed_objects = np.array(row['gazed_objects_names'], dtype=object)
            seg_mask = np.array(row['segment_mask'])
            event_mask = np.array(row['event_mask'], dtype=bool)

            # Keep only valid objects, filter masks accordingly
            valid_idx = [i for i, obj in enumerate(gazed_objects) if obj is not None and str(obj).lower() != 'nan']
            if len(valid_idx) == 0:
                continue

            gazed_objects = gazed_objects[valid_idx]
            seg_mask = seg_mask[valid_idx]
            event_mask = event_mask[valid_idx]

            # Collect objects per segment
            for seg_num, seg_label in segment_labels.items():
                seg_objects = gazed_objects[seg_mask == seg_num]
                seg_objects_all[seg_label].extend(seg_objects.tolist())

            # Collect objects for total E and NE
            E_objects_all.extend(gazed_objects[event_mask].tolist())
            NE_objects_all.extend(gazed_objects[~event_mask].tolist())

        # Helper to get top 3 objects + percentages
        def top3(arr):
            arr = np.array(arr)
            if len(arr) == 0:
                return [], []
            vals, counts = np.unique(arr, return_counts=True)
            sorted_idx = np.argsort(-counts)
            top_objs = vals[sorted_idx[:top_n_gazed]].tolist()
            top_perc = np.round(counts[sorted_idx[:top_n_gazed]] / len(arr) * 100, 2).tolist()
            return top_objs, top_perc

        # Compute top3 per segment
        for seg_label in segment_labels.values():
            objs, perc = top3(seg_objects_all[seg_label])
            stats[f'gazed_top{top_n_gazed}_{seg_label}_obj'] = objs
            stats[f'gazed_top{top_n_gazed}_{seg_label}_perc'] = perc

        # Compute top3 for total E and NE
        objs, perc = top3(E_objects_all)
        stats[f'gazed_top{top_n_gazed}_E_obj'] = objs
        stats[f'gazed_top{top_n_gazed}_E_perc'] = perc

        objs, perc = top3(NE_objects_all)
        stats[f'gazed_top{top_n_gazed}_NE_obj'] = objs
        stats[f'gazed_top{top_n_gazed}_NE_perc'] = perc

        scenario_stats.append(stats)

    return pd.DataFrame(scenario_stats)



# ------------
# --- MAIN ---
# ------------

def main_calculate_statistics(top_n_gazed:int=5, cut_segment7=False, dataframe_save_type='pickle'):
    # --- Load Dataframe ---
    if dataframe_save_type == 'pickle':
        dataframe_save_type = 'pkl'
    data_file = os.path.join(DATA_PATH, f'processed_dataframe.{dataframe_save_type}')
    df = load_dataframe(data_file)
    logger.info(f">>> Processed dataframe loaded from {data_file}")
    logger.info(f">>> (Its Shape: {df.shape})")

    # --- Compute Statistics per session ---
    df_stats = compute_segment_stats_per_session(df, top_n_gazed=top_n_gazed, cut_segment7=cut_segment7)
    logger.info(">>> Segment statistics per session computed.")

    # --- Compute statistics for each scenario ---
    final_df = compute_overall_stats_per_scenario(df_stats)
    logger.info(">>> Overall statistics per scenario computed.")

    # --- Compute gazed objects for overall ---
    df_scenario_gaze = compute_gazed_objects_per_scenario(df, top_n_gazed=top_n_gazed, cut_segment7=cut_segment7)
    logger.info(f">>> Overall top {top_n_gazed} gazed objects computed.")

    # --- Exporting dataframes as pickle files ---
    stats_pickle_file = os.path.join(DATA_PATH, 'statistics_per_session.pkl')
    final_stats_pickle_file = os.path.join(DATA_PATH, 'overall_statistics_per_scenario.pkl')
    gaze_stats_pickle_file = os.path.join(DATA_PATH, 'gazed_objects_per_scenario.pkl')

    save_dataframe_pickle(df_stats, stats_pickle_file)
    save_dataframe_pickle(final_df, final_stats_pickle_file)
    save_dataframe_pickle(df_scenario_gaze, gaze_stats_pickle_file)
    logger.info(f">>> Dataframes saved as pickle files in dir {DATA_PATH}")


if __name__ == "__main__":
    top_n_gazed = 5
    cut_seg7 = False
    dataframe_save_type = 'parquet'  # 'pickle' or 'parquet'
    main_calculate_statistics(top_n_gazed=top_n_gazed, cut_segment7=cut_seg7, dataframe_save_type=dataframe_save_type)
