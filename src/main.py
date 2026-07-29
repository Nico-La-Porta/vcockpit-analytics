import os
import sys
import pickle
import argparse
import logging
import warnings
from typing import List

# Add the project root to the Python path (for standalone execution)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(CURRENT_DIR, "..")
sys.path.append(os.path.abspath(ROOT_PATH))

from src.app_config.config import *
from src.classes.Study import Study
from src.data_ingester import *
from src.extractor import *
from src.utils import logger

from src.analysis.process_study import main_process_study
from src.analysis.calculate_statistics import main_calculate_statistics
from src.analysis.perform_experiments import main_experiments
from src.analysis.perform_experiments_model import main_experiments_model

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_ROOT = PATH['base_root']
DATA_PATH = PATH['data']
STUDY_PICKLE_FILE_PATH = PATH['study_pickle_file_path']

scenario = METADATA['scenario']
users = METADATA['users']
logger.info(f"Scenario: {scenario}, Users: {users}")

if BASE_ROOT not in sys.path:
    sys.path.append(BASE_ROOT)


def extract_data(dirs: dict) -> None:
    if dirs is None or len(dirs) == 0:
        logger.warning("No processable directories found. Exiting extraction.")
        return

    data = None

    for i, (scenario, folders) in enumerate(dirs.items()):
        for folder in folders:
            if not os.path.isdir(os.path.join(DATA_PATH, str(folder))):
                logger.warning(f"\n>>> Folder '{folder}' not present or not a directory\n")
                continue

            logger.info(f"<<Processing folder: {folder}")

            date, time = folder.split('_')[0], folder.split('_')[1]

            logger.info(f"\n>>> Session {date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:]}:\n")

            if i != 0:
                data += MyExtractor(os.path.join(DATA_PATH, str(folder)), save=False)
            else:
                data = MyExtractor(os.path.join(DATA_PATH, str(folder)), save=False)

    pickle_file = os.path.join(DATA_PATH, STUDY_PICKLE_FILE_PATH)

    if not data is None:
        study = Study(data=data.__dict__, scenario=scenario, users=users)

        with open(pickle_file, 'wb') as f:
            pickle.dump(study, f)
            logger.info(f"\tData saved to {pickle_file}")


def run_pipeline(
    logs_dir: str = None,
    cut7: bool = False,
    top_n_gazed: int = 5,
    min_pred: int = 3,
    max_pred: int = 7,
    scenarios: List[str] = ('City', 'Highway'),
    analysis_type: str = 'LMEM',
    clustering: bool = False,
    targets: List[str] = ('mean_awareness', 'mean_safety'),
) -> None:
    """
    Runs the full V-Cockpit analytics pipeline end-to-end:
    ingestion -> processing -> statistics -> experiments (LMEM or RF).
    """
    # --- Ingestion + extraction: builds the Study object ---
    ingester = DataIngester(logs_dir)
    dirs = ingester.get_processable_dirs()
    extract_data(dirs)

    logger.info("")
    logger.info(">>> STUDY OBJECT CREATED")
    logger.info("")

    # --- process_study ---
    main_process_study(cut_segment7=cut7, cut_starting_section=True, update_distraction=True)

    logger.info("")
    logger.info(">>> PROCESSING OF STUDY OBJECT FINISHED")
    logger.info("")

    # --- calculate_statistics ---
    main_calculate_statistics(top_n_gazed=top_n_gazed, cut_segment7=cut7)

    logger.info("")
    logger.info(">>> DATA STATISTICS CALCULATED")
    logger.info("")

    # --- Experiments ---
    scenarios = list(scenarios)
    targets = list(targets)
    always_include = []
    output_dir = os.path.join(BASE_ROOT, 'results_RF') if analysis_type == 'RF' else os.path.join(BASE_ROOT, 'results_LMEM')
    if clustering:
        output_dir += "_clustering"

    if analysis_type == 'LMEM':
        main_experiments(output_dir=output_dir,
                          min_size=min_pred,
                          max_size=max_pred,
                          scenarios=scenarios,
                          targets=targets,
                          always_include=always_include,
                          clustering=clustering)
    elif analysis_type == 'RF':
        main_experiments_model(output_dir=output_dir,
                                min_size=min_pred,
                                max_size=max_pred,
                                scenarios=scenarios,
                                targets=targets,
                                always_include=always_include,
                                clustering=clustering)

    logger.info("")
    logger.info(f">>> {analysis_type} EXPERIMENTS COMPUTED (clustering: {clustering})")
    logger.info(f">>> Output directory: {output_dir}")
    logger.info("")


def arg_parse():
    parser = argparse.ArgumentParser(description="Run the full V-Cockpit analytics pipeline.")
    parser.add_argument('--logs-dir', help='Directory with JSON logs')
    parser.add_argument('--cut7', action="store_true", help='Remove segment 7 from sessions')
    parser.add_argument('--top_n_gazed', type=int, default=5)
    parser.add_argument('--min_pred', type=int, default=3)
    parser.add_argument('--max_pred', type=int, default=7)
    parser.add_argument('--scenarios', nargs='+', default=['City', 'Highway'], choices=['City', 'Highway'], help="Select scenarios from City and Highway")
    parser.add_argument('--analysis_type', type=str, default='LMEM', choices=['RF', 'LMEM'], help="Select analysis type: RF or LMEM")
    parser.add_argument('--clustering', action="store_true")
    TARGETS = ['mean_awareness', 'mean_safety']
    parser.add_argument('--targets', nargs='+', default=TARGETS, choices=TARGETS, help="Select one or more targets from the default list")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    logger.info(f">>>> Input Arguments: cut7 ({args.cut7}), top_n_gazed ({args.top_n_gazed}), min_pred ({args.min_pred}), max_pred ({args.max_pred}), ")
    logger.info(f"     scenarios ({args.scenarios}), targets ({args.targets}), analysis_type ({args.analysis_type}), clustering ({args.clustering})")
    run_pipeline(
        logs_dir=args.logs_dir,
        cut7=args.cut7,
        top_n_gazed=args.top_n_gazed,
        min_pred=args.min_pred,
        max_pred=args.max_pred,
        scenarios=args.scenarios,
        analysis_type=args.analysis_type,
        clustering=args.clustering,
        targets=args.targets,
    )
    print('Done!')
