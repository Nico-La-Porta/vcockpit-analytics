import pandas as pd
import numpy as np
import os
import sys
import json
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import itertools
import argparse
from sklearn.cluster import KMeans
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
from statsmodels.tools.sm_exceptions import ConvergenceWarning
pd.options.mode.chained_assignment = None


DATA_PATH = PATH['data']
BASE_ROOT = PATH['base_root']

# ----------------------
# --- Util functions ---
# ----------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_pred', type=int, default=2)
    parser.add_argument('--max_pred', type=int, default=3) # max_size's value is included
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


# ----------------------------
# --- LMEM Study functions ---
# ----------------------------

def nakagawa_r2_mixedlm(model_result, X_fixed, y):
    """
    Approximate Nakagawa marginal and conditional R2 for statsmodels MixedLM
    model_result: fitted MixedLMResults
    X_fixed: design matrix used for fixed effects (including intercept)
    y: observed outcome (1-d array)
    Returns (R2_marginal, R2_conditional)
    """
    # Predicted fixed-only part
    beta = model_result.params  # includes both fixed and maybe random? MixedLM puts fixed first
    # Extract fixed effect betas only:
    # model_result.model.exog_names gives names; we assume params index matches.
    fixed_names = model_result.model.exog_names
    fixed_beta = model_result.params[fixed_names]
    fitted_fixed = np.dot(X_fixed, fixed_beta)
    var_fixed = np.nanvar(fitted_fixed)

    # random intercept variance
    # model_result.cov_re is covariance matrix of random effects
    try:
        var_random = float(np.diag(model_result.cov_re).sum())
    except Exception:
        # fallback: use estimated random effect variance from random effects
        re = model_result.random_effects
        if len(re) > 0:
            vals = np.array([v[list(v.keys())[0]] if isinstance(v, dict) else v for v in re.values()])
            var_random = np.nanvar(vals)
        else:
            var_random = 0.0

    # residual variance (scale)
    var_resid = float(model_result.scale)  # residual variance

    total = var_fixed + var_random + var_resid
    R2_m = var_fixed / total if total != 0 else np.nan
    R2_c = (var_fixed + var_random) / total if total != 0 else np.nan
    return R2_m, R2_c

def perform_LMEM_study(df_og, scenario, predictors: list, target: str, to_print=False, 
                       categoricals=['gender', 'owns_vehicle', 'vehicle_type', 'segment_type']):
    """
    Used for a single study.
    """
    # Drop rows with no outcome info
    df = df_og.copy(deep=True)
    if target == 'segment_type': 
        df["segment_type"] = df["segment_type"].map({'E': 1, 'NE': 0})
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=[target], how='all')

    # Separate categorical and continuous predictors
    categorical_predictors = [p for p in predictors if p in categoricals]
    cont_predictors = [p for p in predictors if p not in categoricals]

    # Standardize continuous predictors
    scaler = StandardScaler()
    model_df = df.copy().dropna(subset=cont_predictors)
    model_df[cont_predictors] = scaler.fit_transform(model_df[cont_predictors].fillna(model_df[cont_predictors].mean()))

    # Encode categorical predictors
    for pred in categorical_predictors:
        model_df[pred] = model_df[pred].astype('category')

    # Formula
    formula_predictors = " + ".join(cont_predictors + categorical_predictors)
    formula_fixed = f'{target} ~ {formula_predictors}'

    # Filter scenario-specific data
    if scenario.capitalize() == 'City':
        lmem_lane_variance_df = model_df.query("scenario in ['City - calls', 'City - calls - short']")
    elif scenario.capitalize() == 'Highway':
        lmem_lane_variance_df = model_df.query("scenario in ['Highway - music', 'Highway - music - short']")
    else:
        if to_print:
            logger.info("No scenario with such name")
        return

    # Fit model
    md = smf.mixedlm(formula_fixed, data=lmem_lane_variance_df, groups=lmem_lane_variance_df['Subject'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        mres = md.fit(reml=False)

    if to_print:
        logger.info(f"\n=== MixedLM for {target} ({scenario.capitalize()}) ===")
        logger.info(mres.summary())

    # Nakagawa R2 calculation
    exog = mres.model.exog
    y_obs = lmem_lane_variance_df[target].values
    R2_m, R2_c = nakagawa_r2_mixedlm(mres, exog, y_obs)
    # --- R^2 classic & RMSE & MAE ---
    y_pred = mres.fittedvalues.values
    r2_classic = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae = mean_absolute_error(y_obs, y_pred)

    # Coefficients and standardized betas
    coeffs = mres.params.copy()
    std_betas = coeffs.loc[[n for n in coeffs.index if n != 'Intercept']]

    # Confidence intervals for betas
    conf_intervals = mres.conf_int()
    conf_intervals.columns = ['CI_lower', 'CI_upper']
    conf_intervals = conf_intervals.loc[[n for n in conf_intervals.index if n != 'Intercept']]
    # Convert to JSON-safe types
    coeffs_dict = {k: float(v) for k, v in std_betas.to_dict().items()}  # ensure plain floats
    conf_intervals_list = [
        {
            'predictor': str(idx),
            'CI_lower': None if pd.isna(row['CI_lower']) else float(row['CI_lower']),
            'CI_upper': None if pd.isna(row['CI_upper']) else float(row['CI_upper'])
        }
        for idx, row in conf_intervals.iterrows()
    ]

    if to_print:
        logger.info(f"Approx Nakagawa R² (marginal): {R2_m:.3f}, (conditional): {R2_c:.3f}")
        logger.info("\nStandardized betas (fixed effects):")
        logger.info(std_betas)
        logger.info("\nConfidence intervals:")
        logger.info(conf_intervals)

    return {
        'R2_m': float(R2_m),
        'R2_c': float(R2_c),
        'R2_classic': float(r2_classic),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'scale': mres.scale,
        'coefficients': coeffs_dict,
        'conf_intervals': conf_intervals_list
    }, bool(mres.converged)

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
    - exceptions: if the target a key of this dictionary, then its content cannot be considered as predictors.
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
            target_results = []  # store results for this specific target
            
            if cluster_id is not None:
                target_results.append(df_summary_all['Subject'].unique().tolist())

            # Start with all predictors except the target itself
            predictors_options = [p for p in possible_predictors if p != target]

            # Apply exceptions: remove predictors that are forbidden for this target
            if target in exceptions:
                forbidden = exceptions[target]
                predictors_options = [p for p in predictors_options if p not in forbidden]

            # Iterate over all combinations with size limits
            for r in range(min_size_predictors, min(len(predictors_options), max_size_predictors) + 1):
                for pred_combination in itertools.combinations(predictors_options, r):
                
                    if len(always_include)>0: 
                        predictors = list(pred_combination) + always_include
                    else:
                        predictors = list(pred_combination)

                    try:
                        # Run the LMEM study
                        res, converged = perform_LMEM_study(df_summary_all, scenario, predictors, target, to_print=False, categoricals=categoricals)
                        n_experiment += 1
                        if converged:
                            result_entry = {
                                'n_experiment': n_experiment,
                                'n_cluster':cluster_id, 
                                'scenario': scenario,
                                'target': target,
                                'n_predictors': len(predictors),
                                'predictors': predictors,
                                'R2_m': res['R2_m'],
                                'R2_c': res['R2_c'],
                                'R2_classic': res['R2_classic'],
                                'RMSE': res['RMSE'],
                                'MAE': res['MAE'],
                                'scale': res['scale'],
                                'betas': res['coefficients'],
                                'conf_intervals': res['conf_intervals']
                            }

                            target_results.append(result_entry)

                    except Exception:
                        # Skip this combination
                        continue

            # After finishing this target, save its results to JSON 
            # (Will have a results file per target and per scenario, also per cluster if clustering is enabled)
            if target_results:
                if cluster_id is not None:
                    output_path = os.path.join(output_dir, f"{target}_{scenario}_results_cluster_{cluster_id}.json")
                else:                    
                    output_path = os.path.join(output_dir, f"{target}_{scenario}_results.json")
                with open(output_path, "w") as f:
                    json.dump(target_results, f, indent=4)
                logger.info(f"Saved {len(target_results)} results for target '{target}' to {output_path}")


# ------------
# --- MAIN ---
# ------------

def main_experiments(output_dir:str, min_size=4, max_size=8, scenarios=['City', 'Highway'], 
                     targets:list= ['mean_awareness', 'mean_safety'], always_include=['segment_type'], clustering=False, resume=False):
    
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

    if clustering:
        output_dir = os.path.join(BASE_ROOT, 'output/demo_clustering')
    else:
        output_dir = os.path.join(BASE_ROOT, 'output/normal')
    
    logger.info(f">>>> Input Arguments: min_pred ({args.min_pred}), max_pred ({args.max_pred}), Clustering ({clustering})")
    logger.info(f"     always including ({args.always_include}) (len {len(args.always_include)}) , scenarios ({args.scenarios}), targets ({args.targets})")
    logger.info(f"     Output Path -> {output_dir}")
    always_include = []
    main_experiments(output_dir=output_dir, 
                     min_size=args.min_pred, 
                     max_size=args.max_pred,
                     scenarios=args.scenarios,
                     targets=args.targets,
                     always_include=always_include,
                     clustering=clustering)
