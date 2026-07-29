import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.preprocessing import StandardScaler
import itertools
import argparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add the project root to the Python path (for standalone execution)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(os.path.abspath(ROOT_PATH))

from src.app_config.config import *
from src.utils.log_config import logger
from src.analysis.calculate_statistics import load_dataframe

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


DATA_PATH = PATH['data']
BASE_ROOT = PATH['base_root']

# ----------------------
# --- Util functions ---
# ----------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_pred', type=int, default=3)
    parser.add_argument('--max_pred', type=int, default=7) # max_size's value is included
    parser.add_argument('--clustering', action="store_true")
    parser.add_argument('--scenarios', nargs='+', default=['City', 'Highway'], choices=['City', 'Highway'], help="Select scenarios from City and Highway")
    parser.add_argument('--always_include', nargs='*', default=['segment_type'], help='Select columns that should always be included in combinations.')

    TARGETS = ['mean_awareness', 'mean_safety']
    parser.add_argument('--targets', nargs='+', default=TARGETS, choices=TARGETS, help="Select one or more targets from the default list")

    return parser.parse_args()

def safe_get_list(row, col):
    """Return list or np.array for column col in this row; if scalar, wrap it."""
    v = row.get(col, None)
    if v is None:
        return None
    if isinstance(v, (list, np.ndarray, pd.Series)):
        return np.array(v)
    else:
        return np.array([v])

def create_summary_df(df):
    """
    Creates dataframe that is input of the experiment.
    """
    summary_rows = []
    for idx, row in df.iterrows():
        task_ids = safe_get_list(row, 'taskIDs')
        if task_ids is None:
            continue
        n_events = len(task_ids)

        # time-series arrays
        arr_distraction = safe_get_list(row, 'distraction')
        arr_focus = safe_get_list(row, 'focus')
        arr_lane = safe_get_list(row, 'lane')
        arr_speed = safe_get_list(row, 'vehicle_speed')
        arr_safety_mask = safe_get_list(row, 'vehicle_safety_distance_mask')
        arr_ppg = safe_get_list(row, 'ppg')
        arr_involvement = safe_get_list(row, 'involvement')
        arr_awareness = safe_get_list(row, 'awareness')
        arr_safety = safe_get_list(row, 'safety')
        arr_gaze_category = safe_get_list(row, 'gaze_category_mask')
        arr_offset = safe_get_list(row, 'vehicle_straight_driving_offset')

        # event start/end indices
        starts = safe_get_list(row, 'event_start_scenario')
        ends = safe_get_list(row, 'event_end_scenario')

        # reaction/execution times
        event_reaction = safe_get_list(row, 'event_reaction_time')
        event_execution = safe_get_list(row, 'event_execution_time')

        # session bounds
        session_start = 0
        session_end = len(arr_distraction) if arr_distraction is not None else 0

        # Build slices: NE and Tasks
        slices = []
        if n_events == 0:
            slices.append({'start': session_start, 'end': session_end, 'type': 'NE', 'task_idx': np.nan, 'taskID': None})
        else:
            # NE before first event
            if starts[0] > session_start:
                slices.append({'start': session_start, 'end': int(starts[0]), 'type': 'NE', 'task_idx': np.nan, 'taskID': None})
            for i in range(n_events):
                start_i = int(starts[i])
                end_i = int(ends[i])
                # Task slice
                slices.append({'start': start_i, 'end': end_i, 'type': 'E', 'task_idx': i, 'taskID': task_ids[i]})
                # NE between events
                if i < n_events - 1 and int(starts[i+1]) > end_i:
                    slices.append({'start': end_i, 'end': int(starts[i+1]), 'type': 'NE', 'task_idx': np.nan, 'taskID': None})
            # NE after last event
            if ends[-1] < session_end:
                slices.append({'start': int(ends[-1]), 'end': session_end, 'type': 'NE', 'task_idx': np.nan, 'taskID': None})

        # summarize each slice
        for s in slices:
            slc = slice(s['start'], s['end'])

            def sliced_stat(arr, func=np.nanmean):
                if arr is None or len(arr) == 0:
                    return np.nan
                seg = np.array(arr[slc])
                if seg.size == 0:
                    return np.nan
                return func(seg[np.isfinite(seg)])

            def sliced_stat_2(arr, func=np.nanmean):
                if arr is None or len(arr) == 0:
                    return np.nan
                try:
                    slc_new = slice(round(s['start']/100), round(s['end']/100))
                    seg = np.array(arr[slc_new])
                    if seg.size == 0:
                        return np.nan
                    return func(seg[np.isfinite(seg)])
                except Exception:
                    return func(np.array(arr)[np.isfinite(arr)])

            mean_distraction = sliced_stat(arr_distraction)
            distraction_std = sliced_stat(arr_distraction, np.nanstd)
            mean_focus = sliced_stat(arr_focus)
            focus_std = sliced_stat(arr_focus, np.nanstd)
            lane_variance = sliced_stat(arr_lane, np.nanvar)
            speed_variance = sliced_stat(arr_speed, np.nanvar)
            mean_speed = sliced_stat(arr_speed)
            mean_awareness = sliced_stat_2(arr_awareness)
            awareness_std = sliced_stat_2(arr_awareness, np.nanstd)
            mean_safety = sliced_stat_2(arr_safety)
            safety_std = sliced_stat_2(arr_safety, np.nanstd)
            mean_safety_distance = sliced_stat(arr_safety_mask)
            mean_involvement = sliced_stat(arr_involvement)
            offset_mean = sliced_stat(arr_offset)
            offset_var = sliced_stat(arr_offset, np.nanvar)
            category_true_perc = sliced_stat(arr_gaze_category, lambda x: round(np.count_nonzero(x) / np.count_nonzero(~np.isnan(x)) * 100, 2))

            safety_violations = 0
            if arr_safety_mask is not None:
                safety_violations = np.nansum(np.array(arr_safety_mask[slc]) > 0)

            # reaction/execution (only for Tasks)
            reaction_time = np.nan
            execution_time = np.nan
            if s['type'] == 'E' and event_reaction is not None and len(event_reaction) > s['task_idx']:
                reaction_time = float(event_reaction[s['task_idx']])
            if s['type'] == 'E' and event_execution is not None and len(event_execution) > s['task_idx']:
                execution_time = float(event_execution[s['task_idx']])

            # subject-level info
            subj = row.get('Subject', row.get('user_id', np.nan))
            age = row.get('age', np.nan)
            gender = row.get('gender', np.nan)
            bmi = row.get('bmi', np.nan)
            prev_exp = row.get('vr_exp', np.nan)
            driving_exp = row.get('driving_exp', np.nan)
            driving_freq = row.get('driving_freq', np.nan)
            vehicle_owned = row.get('owns_vehicle', np.nan)
            vehicle_type = row.get('vehicle_type', np.nan)

            summary_rows.append({
                'Subject': subj,
                'task_idx': s['task_idx'],
                'taskID': s['taskID'],
                'segment_type': s['type'],  # 'E' or 'NE'
                'scenario': row.get('scenario', np.nan),
                'mean_distraction': mean_distraction,
                'distraction_std': distraction_std,
                'mean_focus': mean_focus,
                'focus_std': focus_std,
                'mean_awareness': mean_awareness,
                'awareness_std': awareness_std,
                'mean_safety': mean_safety,
                'safety_std': safety_std,
                'mean_safety_distance': mean_safety_distance,
                'mean_involvement': mean_involvement,
                'lane_variance': lane_variance,
                'speed_variance': speed_variance,
                'mean_speed': mean_speed,
                'safety_violations': safety_violations,
                'offset_mean': offset_mean,
                'offset_var': offset_var,
                'gaze_category_true_perc': category_true_perc,
                'reaction_time': reaction_time,
                'execution_time': execution_time,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'vr_exp': prev_exp,
                'driving_exp': driving_exp,
                'driving_freq': driving_freq,
                'owns_vehicle': vehicle_owned,
                'vehicle_type': vehicle_type,
            })

    df_summary_all = pd.DataFrame(summary_rows)
    return df_summary_all

def get_max_n_experiment(filepath):
    """Return the max 'n_experiment' value found in an existing results file, or 0."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return 0
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0
    max_n = 0
    for entry in data:
        if isinstance(entry, dict) and 'n_experiment' in entry:
            max_n = max(max_n, entry['n_experiment'])
    return max_n

# JSON-related writing functions
def get_completed_predictor_sets(filepath):
    """
    Read an existing results JSON file and return the set of predictor
    combinations (as frozensets) that already have a saved result.
    Returns an empty set if the file doesn't exist or is empty/invalid.
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return set()

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()

    done = set()
    for entry in data:
        if isinstance(entry, dict) and 'predictors' in entry:
            done.add(frozenset(entry['predictors']))
    return done

def append_json_array(filepath, entry, indent=4):
    """
    Append `entry` to a JSON file containing a top-level JSON array,
    creating the file if it doesn't exist. Does NOT read the whole
    file into memory - only scans backward a few bytes to find ']'.
    """
    entry_str = json.dumps(entry, indent=indent)

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            f.write('[\n' + entry_str + '\n]')
        return

    with open(filepath, 'rb+') as f:
        f.seek(0, os.SEEK_END)
        filesize = f.tell()

        pos = filesize - 1
        while pos >= 0:
            f.seek(pos)
            c = f.read(1)
            if c == b']':
                break
            pos -= 1
        if pos < 0:
            raise ValueError(f"{filepath} does not look like a JSON array")
        close_bracket_pos = pos

        p = close_bracket_pos - 1
        last_char = None
        while p >= 0:
            f.seek(p)
            c = f.read(1)
            if c not in b' \t\r\n':
                last_char = c
                break
            p -= 1
        is_empty = (last_char == b'[')

        f.seek(close_bracket_pos)
        f.truncate()

        payload = (entry_str + '\n]') if is_empty else (',\n' + entry_str + '\n]')
        f.write(payload.encode('utf-8'))

def init_json_array(filepath):
    """Start (or reset) a file as an empty JSON array."""
    with open(filepath, 'w') as f:
        f.write('[]')

# ---------------------------
# --- RF Study functions ---
# ---------------------------
def compute_icc(df, scenario, target):
    """
    Calcola una stima approssimata dell'ICC (quota di varianza tra soggetti)
    per un dato target, filtrato per scenario.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target], how='all')

    if scenario.capitalize() == 'City':
        scenario_df = df.query("scenario in ['City - calls', 'City - calls - short']")
    elif scenario.capitalize() == 'Highway':
        scenario_df = df.query("scenario in ['Highway - music', 'Highway - music - short']")
    else:
        return None

    if scenario_df.empty or scenario_df['Subject'].nunique() < 2:
        return None

    subject_means = scenario_df.groupby('Subject')[target].transform('mean')
    grand_mean = scenario_df[target].mean()

    ss_between = np.sum((subject_means - grand_mean) ** 2)
    ss_total = np.sum((scenario_df[target] - grand_mean) ** 2)

    if ss_total == 0:
        return None

    return float(ss_between / ss_total)

def perform_RF_study(df_og, scenario, predictors: list, target: str, to_print=True,
                               categoricals=['gender', 'owns_vehicle', 'vehicle_type', 'segment_type']):
    
    df = df_og[predictors + [target, 'Subject', 'scenario']].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target], how='all')

    # Filter scenario
    if scenario.capitalize() == 'City':
        model_df = df.query("scenario in ['City - calls', 'City - calls - short']").copy()
    elif scenario.capitalize() == 'Highway':
        model_df = df.query("scenario in ['Highway - music', 'Highway - music - short']").copy()
    else:
        if to_print:
            print("No scenario with such name")
        return None

    model_df = model_df.dropna(subset=predictors)
    if model_df.empty:
        return None

    categorical_predictors = [p for p in predictors if p in categoricals]
    cont_predictors = [p for p in predictors if p not in categoricals]

    # Scale continuous predictors
    scaler = StandardScaler()
    if cont_predictors:
        model_df[cont_predictors] = scaler.fit_transform(model_df[cont_predictors])

    # One-hot encode categoricals
    model_df = pd.get_dummies(model_df, columns=categorical_predictors, drop_first=True)

    X = model_df.drop(columns=[target, 'Subject', 'scenario'])
    y = model_df[target]

    if len(X) == 0:
        return None

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    r2_classic = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))

    # Feature importance
    feature_importance = dict(zip(X.columns, rf.feature_importances_))
    feature_importance = {k: float(v) for k, v in feature_importance.items()}

    # Permutation importance 
    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    permutation_importance_dict = {k: float(v) for k, v in zip(X.columns, perm.importances_mean)}

    if to_print:
        print(f"\n=== RF In-Sample for {target} ({scenario}) ===")
        print(f"R2_classic: {r2_classic:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")

    return {
        'R2_classic': r2_classic,
        'RMSE': rmse,
        'MAE': mae,
        'feature_importance': feature_importance,
        'permutation_importance': permutation_importance_dict,
    }

def experiment_combinations(df_summary_all, 
                            output_dir = os.path.join(BASE_ROOT, 'results_experiments'), 
                            always_include:list=['segment_type'], 
                            min_size_predictors:int = 3, 
                            max_size_predictors:int = 7, 
                            scenarios:list = ['City', 'Highway'],
                            possible_predictors:list = ['mean_distraction', 'mean_focus', 'mean_safety_distance', 'lane_variance', 
                                                   'speed_variance', 'safety_violations', 'offset_var','bmi', 'vr_exp', 'driving_exp', 'driving_freq', 'owns_vehicle', 'vehicle_type'],
                            possible_targets:list= ['mean_awareness', 'mean_safety'],
                            exceptions:dict={'mean_awareness': ['mean_safety', 'gaze_category_true_perc'], 'mean_safety': ['mean_awareness', 'gaze_category_true_perc'], 'segment_type': ['segment_number', 'mean_awareness', 'mean_safety', 'owns_vehicle', 'segment_type']},
                            cluster_id=None, resume=False):
    """
    - df: input dataframe.
    - always_include: items in this list are always included in the combination.
    - min/max_size_predictors: min/max number of predictors in the combination.
    - scenarios: scenarios to iterate.
    - possible_predictors: list of predictor names to combine.
    - possible_targets: targets to iterate during the combinations.
    - exceptions: if the target is a key of this dictionary, then its content cannot be considered as predictors.
    - cluster_id: if not None, it indicates the cluster number (as the function is run for each cluster if "clustering" is True).
    """
    categoricals = ['gender', 'owns_vehicle', 'vehicle_type', 'segment_type']
    n_experiment = 1
    logger.info(f"ALWAYS INCLUDE LEN {len(always_include)}")
    if len(always_include)==0: # if we do not always include it, keep it as a combinable predictor.
        possible_predictors.append('segment_type')

    logger.info(f">>> Possible predictors (before): {possible_predictors}")
    possible_predictors = list(dict.fromkeys(possible_predictors))
    logger.info(f">>> CLUSTER N {cluster_id}")
        
    for scenario in scenarios:
        logger.info(f"Scenario: {scenario}")
        for target in possible_targets:
            logger.info(f"Target: {target}")
            logger.info(f"Experiments so far: {n_experiment-1}")

            icc = compute_icc(df_summary_all, scenario, target)
            logger.info(f"ICC (quota varianza tra soggetti) per {target}/{scenario}: {icc}")


            if cluster_id is not None:
                output_path = os.path.join(output_dir, f"{target}_{scenario}_results_cluster_{cluster_id}.json")
            else:
                output_path = os.path.join(output_dir, f"{target}_{scenario}_results.json")
            
            file_exists_nonempty = os.path.exists(output_path) and os.path.getsize(output_path) > 0

            if resume and file_exists_nonempty:
                completed = get_completed_predictor_sets(output_path)
                n_experiment = get_max_n_experiment(output_path)  # continue numbering
                logger.info(f"Resuming: {len(completed)} experiments already done for {target}/{scenario}")
                need_user_list = False
            else:
                completed = set()
                init_json_array(output_path)
                need_user_list = cluster_id is not None

            n_saved = 0

            if need_user_list:
                append_json_array(output_path, df_summary_all['Subject'].unique().tolist())
                n_saved += 1

            predictors_options = [p for p in possible_predictors if p != target]
            if target in exceptions:
                forbidden = exceptions[target]
                predictors_options = [p for p in predictors_options if p not in forbidden]

            for r in range(min_size_predictors, min(len(predictors_options), max_size_predictors) + 1):
                for pred_combination in itertools.combinations(predictors_options, r):

                    if len(always_include) > 0:
                        predictors = list(pred_combination) + always_include
                    else:
                        predictors = list(pred_combination)

                    if resume and frozenset(predictors) in completed:
                        continue  # already done, skip

                    try:
                        res = perform_RF_study(df_summary_all, scenario, predictors, target, to_print=False, categoricals=categoricals)
                        n_experiment += 1
                        if res is not None:
                            result_entry = {
                                'n_experiment': n_experiment,
                                'scenario': scenario,
                                'target': target,
                                'icc': icc,
                                'n_predictors': len(predictors),
                                'predictors': predictors,

                                'R2_classic': res['R2_classic'],
                                'RMSE': res['RMSE'],
                                'MAE': res['MAE'],

                                'feature_importance': res['feature_importance'],
                                'permutation_importance': res['permutation_importance'],
                            }
                            append_json_array(output_path, result_entry)
                            n_saved += 1

                    except Exception as e:
                        logger.info(f"Error with predictors {predictors} and target {target}: {e}")
                        continue

            logger.info(f"Saved {n_saved} new results for target '{target}' to {output_path} (resume={resume})")



# ------------
# --- MAIN ---
# ------------

def main_experiments_model(output_dir:str, min_size=4, max_size=8, scenarios=['City', 'Highway'], 
                     targets:list= ['mean_awareness', 'mean_safety'], always_include=['segment_type'], 
                     clustering=False, resume=False):
    
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Dataframe ---
    data_file_path = os.path.join(DATA_PATH, 'processed_dataframe.parquet')
    df = load_dataframe(data_file_path, file_type='parquet')
    logger.info(f">>> Processed dataframe loaded from {data_file_path} .")
    logger.info(f">>> (its Shape: {df.shape})")

    # --- Creating dataframe for experiments ---
    df_summary_all = create_summary_df(df)
    logger.info(f">>> Processed summary dataframe.")
    logger.info(f">>> (its Shape: {df_summary_all.shape})")

    
    # --- Perform experiments ---
    if clustering:
        user_df = df_summary_all.groupby('Subject').mean(numeric_only=True).reset_index()
        features = ['driving_exp', 'driving_freq'] 
        X = user_df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        user_df['cluster'] = kmeans.fit_predict(X_scaled)
        cluster_users = user_df.groupby('cluster')['Subject'].apply(list)
        for cluster_id, users in cluster_users.items():
            subset = df_summary_all[df_summary_all['Subject'].isin(users)]
            logger.info(f">>> Processing Cluster {cluster_id} (Users {users})")
            experiment_combinations(df_summary_all=subset,
                            output_dir=output_dir,
                            always_include=always_include,
                            min_size_predictors=min_size,
                            max_size_predictors=max_size,
                            scenarios=scenarios,
                            possible_targets=targets,
                            cluster_id=cluster_id,
                            resume=resume)


    else: # Normal experiments
        logger.info(f">>> Processing All Data")
        experiment_combinations(df_summary_all=df_summary_all,
                                output_dir=output_dir,
                                always_include=always_include,
                                min_size_predictors=min_size,
                                max_size_predictors=max_size,
                                scenarios=scenarios,
                                possible_targets=targets, 
                                resume=resume)
    logger.info(">>> Experiments Completed!")


if __name__ == "__main__":
    args= parse_args()
    clustering = args.clustering
    resume=True

    if clustering:
        output_dir = os.path.join(BASE_ROOT, 'output/model_clustering_insample')
    else:
        output_dir = os.path.join(BASE_ROOT, 'output/model')
    
    logger.info(f">>>> Input Arguments: min_pred ({args.min_pred}), max_pred ({args.max_pred}), Clustering ({clustering})")
    logger.info(f"     always including ({args.always_include}) (len {len(args.always_include)}) , scenarios ({args.scenarios}), targets ({args.targets})")
    logger.info(f"     Output Path -> {output_dir}")
    always_include = []
    main_experiments_model(output_dir=output_dir, 
                     min_size=args.min_pred, 
                     max_size=args.max_pred,
                     scenarios=args.scenarios,
                     targets=args.targets,
                     always_include=always_include,
                     clustering=clustering,
                     resume=resume)
