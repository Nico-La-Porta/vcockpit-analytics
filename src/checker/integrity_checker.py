from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from src.utils.log_config import logger

TASKS = 5

MANDATORY_FILES = [
    "vr_session.ini",
    "vr_scenario.json",
    "vr_scenario_task_sensor.csv",
    "vr_gaze_detection.csv",
    "analytics_gaze_object.csv",
    "vr_vehicle_speed.csv",
    "vr_bitalino.csv",
    "vr_distance.csv",
    "vr_lane_sensor.csv",
]

# -------------------- SCENARIO CHECKER (<user>/<scenario>/files..) --------------------

class ScenarioChecker:
    """
    Performs checks on the scenario folder:
      <user_id>/<scenario>/
          file1...
          file2...
    """

    def __init__(self, scenario_path: Path):
        self.scenario_path = scenario_path

        self.files: list[Path] = []
        self.file_names: set[str] = set()

        # results
        self.number_of_files: int = 0
        self.number_of_missing_files: int = -1
        self.missing_files: list[str] = [] # based only on mandatory files
        self.empty_files: list[str] = []

        self.scenario: str | None = None

        self.all_events_present: bool = False
        self.number_of_events: int = -1

        self.processing_duration_ms: int = 0

        self._read_folder()
        self._start = time.perf_counter()

    def _read_folder(self) -> None:
        self.files = [p for p in self.scenario_path.iterdir() if p.is_file()]
        self.file_names = {p.name for p in self.files}

    def _iter_files(self) -> Iterable[Path]:
        """Iterates ONLY direct child files of <folder>/."""
        for p in self.scenario_path.iterdir():
            if p.is_file():
                yield p

    def run_pre_extraction_checks(self) -> None:
        self.check_number_of_files()
        self.check_missing_files()
        self.check_number_of_events()
        self.check_empty_files()
        self.get_scenario_name()
        self.derive_processing_duration_ms()

    # ---------- checks ----------

    def check_number_of_files(self) -> None:
        self.number_of_files = len(self.files)

    def check_missing_files(self) -> None:
        self.missing_files = [name for name in MANDATORY_FILES if name not in self.file_names]
        self.number_of_missing_files = len(self.missing_files)

    def check_number_of_events(self) -> None:
        target = self.scenario_path / "vr_scenario_task_sensor.csv"
        if not target.exists():
            return

        with target.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            row_count = sum(1 for _ in reader)

        self.number_of_events = row_count
        self.all_events_present = (row_count >= TASKS)

    def check_empty_files(self) -> None:
        """
        Marks files as "empty" using type-specific rules:

        - CSV: empty if it has no rows after header (or file has no rows at all)
        - JSON: empty if file is 0 bytes OR cannot be parsed OR parsed content is empty
            (empty dict {}, empty list [], empty string)
        - INI:  empty if file is 0 bytes OR has no non-empty, non-comment lines

        Notes:
        - We also track empty mandatory files in self.empty_mandatory_files
        """
        for file in self._iter_files():
            suffix = file.suffix.lower()

            is_empty = False

            # --- CSV ---
            if suffix == ".csv":
                try:
                    with file.open(encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)

                        # If there's no header row at all, treat as empty
                        header = next(reader, None)
                        if header is None:
                            is_empty = True
                        else:
                            # no rows after header -> empty
                            is_empty = not any(True for _ in reader)
                except Exception:
                    # if unreadable -> treat as empty for safety
                    is_empty = True

            # --- JSON ---
            elif suffix == ".json":
                try:
                    if file.stat().st_size == 0:
                        is_empty = True
                    else:
                        with file.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        # consider empty containers / strings as empty
                        if data is None:
                            is_empty = True
                        elif isinstance(data, (dict, list, str)) and len(data) == 0:
                            is_empty = True
                except Exception:
                    # invalid json -> treat as empty (or "bad") for your pipeline
                    is_empty = True

            # --- INI ---
            elif suffix == ".ini":
                try:
                    if file.stat().st_size == 0:
                        is_empty = True
                    else:
                        with file.open("r", encoding="utf-8") as f:
                            # non-empty line that isn't comment
                            has_content = any(
                                (line.strip() and not line.strip().startswith(("#", ";")))
                                for line in f
                            )
                        is_empty = not has_content
                except Exception:
                    is_empty = True

            # --- OTHER FILE TYPES ---
            else:
                # 0 bytes as empty
                try:
                    is_empty = (file.stat().st_size == 0)
                except Exception:
                    is_empty = True

            if is_empty:
                if file.name not in self.empty_files:
                    self.empty_files.append(file.name)


    def get_scenario_name(self) -> None:
        """
        Reads vr_scenario.json and extracts a scenario name.
        """
        path = self.scenario_path / "vr_scenario.json"
        if not path.exists():
            self.scenario = self.scenario_path.name
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.scenario = data.get("title") or data.get("scenario") or self.scenario_path.name
            else:
                self.scenario = self.scenario_path.name
        except Exception:
            self.scenario = self.scenario_path.name

    def derive_processing_duration_ms(self) -> None:
        elapsed_s = time.perf_counter() - self._start
        self.processing_duration_ms = int(elapsed_s * 1000)

    def is_processable(self) -> bool:
        if self.missing_files:
            return False
        mandatory_empty = [f for f in self.empty_files if f in MANDATORY_FILES]
        if mandatory_empty:
            return False
        if not self.all_events_present:
            return False
        return True

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario": to_snake_case(self.scenario),
            "processing_duration_ms": self.processing_duration_ms,
            "number_of_files": self.number_of_files,
            "missing_files": self.missing_files,
            "empty_files": self.empty_files,
            "tasks_completed": self.number_of_events if self.number_of_events >= 0 else None,
            "is_processable": self.is_processable(),
            "folder_name": self.scenario_path.name,
        }

def to_snake_case(s: str) -> str:
    # remove symobls
    s = re.sub(r'[^0-9a-zA-Z\s]', '', s)
    # substitute spaces with underscore
    s = re.sub(r'\s+', '_', s)
    
    return s.lower()



@dataclass
class UserChecker:
    """
    Updates sessions in an existing user JSON using folders from sessions_dir.
    """
    sessions_dir: Path      # all sessions for all users
    data_root: Path         # e.g., sessions_dir.parent
    logs_folder_name: Path  # name of logs_json folder

    def run(self, user_json_file) -> Path:
        data = self._load_existing_json(user_json_file)
        isChecked = data.get("checked")
        if isChecked:
            logger.info(f"Skipping {user_json_file} ALREADY checked")
            return

        # get current sessions
        sessions = data.get("sessions", [])

        # map all folders in sessions_dir
        all_folders = {p.name: p for p in self.sessions_dir.iterdir() if p.is_dir()}

        for session_obj in sessions:
            folder_name = session_obj.get("folder_name")
            if not folder_name:
                continue
            folder_path = all_folders.get(folder_name)
            if folder_path is None:
                logger.info(f"Folder {folder_name} not found in {self.sessions_dir}")
                continue
            checker = ScenarioChecker(folder_path)
            checker.run_pre_extraction_checks()
            session_obj.update(checker.to_payload())

        data["sessions"] = sessions
        data['checked'] = True

        self._write_json_atomic(user_json_file, data)
        return user_json_file
    
    def _load_existing_json(self, json_file_path: Path) -> dict[str, Any]:
        if json_file_path.exists() and json_file_path.is_file():
            with json_file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        else:
            json_file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
        return data

    def _write_json_atomic(self, json_file_path: Path, data: dict[str, Any]) -> None:
        tmp_path = json_file_path.with_name(f".{json_file_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        tmp_path.replace(json_file_path)

@dataclass
class UsersBatchChecker:
    sessions_dir: Path
    logs_dir: Path

    def run(self) -> list[Path]:
        if not self.sessions_dir.exists() or not self.sessions_dir.is_dir():
            logger.error(f"sessions_dir does not exist or is not a directory: {self.sessions_dir}")

        updated_files: list[Path] = []
        # iterate all JSONs in logs_dir
        for user_json_file in self.logs_dir.glob("*.json"):
            checker = UserChecker(
                sessions_dir=self.sessions_dir,
                data_root=self.sessions_dir.parent,
                logs_folder_name=self.logs_dir.name
            )
            updated_files.append(checker.run(user_json_file))
        return updated_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UsersBatchChecker")

    parser.add_argument(
        "--sessions-dir",
        required=True,
        type=Path,
        help="Path to raw sessions",
    )

    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Logs directory containing user JSONs",
    )

    args = parser.parse_args()

    batch = UsersBatchChecker(
        sessions_dir=args.sessions_dir,
        logs_dir=args.logs_dir
    )

    updated_paths = batch.run()
    for p in updated_paths:
        if p is not None:
            print(p, "- checked")