import os
import json
from typing import List
from pathlib import Path

POSSIBLE_SCENARIOS = ["city_calls_short", "city_calls_long", "highway_music_short", "highway_music_long"]

JSON_LOGS_DIR = Path(os.getcwd()) / 'data' / 'logs_json'

class DataIngester:
    def __init__(self, json_logs_dir):
        self._json_logs_dir = Path(json_logs_dir)
        self._ingest_data()

    def _ingest_data(self):
        processable_sessions = dict()
        processable_sessions["city_calls_short"] = []
        processable_sessions["city_calls"] = []
        processable_sessions["highway_music_short"] = []
        processable_sessions["highway_music"] = []

        for file_path in self.json_logs_dir.glob('*.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
                for s in data.get('sessions', []):
                    scenario = s.get('scenario')
                    if s["is_processable"]:
                        processable_sessions[scenario].append(s["folder_name"])

        self.processable_sessions = processable_sessions
    
    def get_processable_dirs(self, scenario: str | List[str] = "all") -> dict:
        if scenario == "all":
            return self.processable_sessions
        if isinstance(scenario, str) and scenario in POSSIBLE_SCENARIOS:
            return self.processable_sessions.get(scenario, [])
        elif isinstance(scenario, list):
            return {scen: self.processable_sessions.get(scen, []) for scen in scenario if scen in POSSIBLE_SCENARIOS}
        else:
            raise ValueError("Scenario must be a string or a list of strings. Select from: " + ", ".join(POSSIBLE_SCENARIOS))

    
    # Getter
    @property
    def json_logs_dir(self):
        return self._json_logs_dir
    
    # Setter
    @json_logs_dir.setter
    def json_logs_dir(self, new_dir):
        self._json_logs_dir = Path(new_dir)
    
    def __str__(self):
        return f"DataIngester(json_logs_dir={self.json_logs_dir})"
    
# Example usage:
if __name__ == "__main__":
    ingester = DataIngester(JSON_LOGS_DIR)
    dirs = ingester.get_processable_dirs(["city_calls_short"]) # , "city_calls_long"])
    print(f"Ingested JSON files.")