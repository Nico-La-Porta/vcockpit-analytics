import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from src.extractor.Extractor import MyExtractor
from src.utils.utils import extract_session_start_end_timestamp

@pytest.fixture
def extractor():
    # MyExtractor.__init__ eagerly reads session files from disk; patch out the
    # file-touching steps so a bare instance can be built for unit testing.
    with patch('src.extractor.Extractor.extract_session_start_end_timestamp', return_value=(0, 0)), \
         patch.object(MyExtractor, 'get_task_id'), \
         patch.object(MyExtractor, 'extract_event'), \
         patch.object(MyExtractor, 'extract_user_data'), \
         patch.object(MyExtractor, 'reaction_execution_time'), \
         patch.object(MyExtractor, 'traffic_light_check'):
        return MyExtractor("test_folder")

def test_init(extractor):
    assert extractor.folder_path == "test_folder"
    assert extractor.fs == 100
    assert extractor.device == "Bitalino"

@patch('builtins.open', new_callable=mock_open, read_data='start_timestamp=1000\nend_timestamp=2000\n')
def test_extract_session_start_end_timestamp(mock_file):
    start, end = extract_session_start_end_timestamp("test_folder")
    assert start == 1000
    assert end == 2000

@patch('pandas.read_csv')
def test_get_task_id(mock_read_csv, extractor):
    mock_read_csv.return_value = pd.DataFrame({'TaskID': ['TASK1', 'TASK2']})
    with patch('yaml.load', return_value={'taskIDs': ['TASK1', 'TASK2']}):
        extractor.get_task_id()
        assert extractor.taskIDs[-1] == ['TASK1', 'TASK2']

@patch('pandas.read_csv')
def test_extract_event(mock_read_csv, extractor):
    mock_read_csv.return_value = pd.DataFrame({
        'startEventMs': [1000, 2000],
        'endEventMs': [1500, 2500]
    })
    extractor.session_start_timestamp = 500
    extractor.extract_event()
    np.testing.assert_array_equal(extractor.event_start_scenario[-1], [500, 1500])
    np.testing.assert_array_equal(extractor.event_end_scenario[-1], [1000, 2000])

@patch('pandas.read_csv')
def test_reaction_execution_time(mock_read_csv, extractor):
    # reaction_execution_time reads, in order: (1) vr_gaze_detection.csv - only rows
    # gazing at "SCREEN_Screen_000" matter, matched against (2) vr_scenario_task_sensor.csv,
    # whose TaskID rows are positionally aligned with event_start_scenario/event_end_scenario.
    mock_read_csv.side_effect = [
        pd.DataFrame({'gazedName': ['SCREEN_Screen_000'], 'startEventMs': [1050]}),
        pd.DataFrame({'TaskID': ['CALL_0']}),
    ]
    extractor.taskIDs = [['CALL_0']]
    extractor.event_start_scenario = [np.array([1000])]
    extractor.event_end_scenario = [np.array([1500])]
    extractor.reaction_execution_time()
    assert extractor.event_start_subject[-1] == [1050]
    np.testing.assert_array_equal(extractor.event_reaction_time[-1], [50])
    np.testing.assert_array_equal(extractor.event_execution_time[-1], [450])

@patch('pandas.read_csv')
def test_traffic_light_check(mock_read_csv, extractor):
    mock_read_csv.return_value = pd.DataFrame({
        'TrafficSignalColor': [" RED"],
        'endEventMs': [1000],
        'Velocity[m/s]': [10.0]
    })
    extractor.traffic_light_check()
    assert extractor.traffic_light_crossed_red == 1
    assert extractor.traffic_light_time == 1000
    assert extractor.traffic_light_velocity == 10.0

@patch('pandas.read_csv')
def test_extract_vehicle_speed(mock_read_csv, extractor):
    mock_read_csv.return_value = pd.DataFrame({
        'startEventMs': [0, 1000],
        ' VehicleSpeed [m/s]': [10, 20]
    })
    extractor.ppg = pd.Series([0] * 2000)
    extractor.extract_vehicle_speed()
    assert len(extractor.vehicle_speed) == 2000
    assert pytest.approx(extractor.vehicle_speed[0], 0.1) == 36.0  # 10 m/s = 36 km/h

def test_extract_involvement(extractor):
    # Needs enough peaks (>=4) for the cubic interpolation in extract_involvement,
    # hence a longer/denser signal than the other extract_* tests use.
    n_samples = 4000
    extractor.ppg = pd.Series(np.sin(np.linspace(0, 40*np.pi, n_samples)))
    extractor.time_ms = np.arange(n_samples)
    extractor.extract_involvement()
    assert len(extractor.involvement) == n_samples

@patch('pandas.read_csv')
def test_extract_focus(mock_read_csv, extractor):
    mock_read_csv.return_value = pd.DataFrame({
        'gazedName': ['windshield', 'dashboard'],
        'deltaMs': [500, 500]
    })
    extractor.ppg = pd.Series([0] * 1000)
    with patch('json.load', return_value={'windshield': 1, 'dashboard': 2}):
        extractor.extract_focus()
    assert len(extractor.focus) == 1000

@patch('pandas.read_csv')
def test_extract_gaze_track(mock_read_csv, extractor):
    # extract_gaze_track maps 'lookThrough'/'category'/'gazedName' through the real
    # mapping JSONs in src/utils/ (not mocked - they're small, static, repo-tracked
    # files), keyed by upper-case labels such as "WINDSCREEN".
    mock_read_csv.return_value = pd.DataFrame({
        'lookThrough': ['WINDSCREEN', 'WINDSCREEN'],
        'category': ['WINDSCREEN', 'WINDSCREEN'],
        'gazedName': ['WINDSCREEN', 'WINDSCREEN'],
        'endEventMs': [500, 1000],
    })
    extractor.ppg = pd.Series([0] * 1000)
    extractor.extract_gaze_track()
    assert len(extractor.gaze_through_array) == 1000
    assert len(extractor.category) == 1000
    assert len(extractor.gazed_objects) == 1000
    assert extractor.gaze_through_array[0] == 1.0  # WINDSCREEN -> look_through_map == 1
    assert extractor.category[0] == 1.0            # WINDSCREEN -> category_map == 1
    assert extractor.gazed_objects[0] == 1.0        # WINDSCREEN -> objects_map == 1

def test_extract_distraction_sustained_off_task_gaze_increases(extractor):
    # extract_distraction is now driven by gazed_objects (+ age/driving_exp for the
    # tau adjustment), not by focus/category/vehicle_speed/involvement.
    extractor.age = [30]
    extractor.driving_exp = ["more than 10 years"]
    # SCREEN_SCREEN_000 (id 10) is not in the task-relevant set {1, 3, 9, 13}, so
    # sustained gaze on it should accumulate a growing (non-negative) distraction.
    extractor.gazed_objects = np.array([10] * 20, dtype=float)

    extractor.extract_distraction(smooth_win=1)

    assert len(extractor.distraction) == 20
    assert np.all(extractor.distraction >= 0.0) and np.all(extractor.distraction <= 100.0)
    assert extractor.distraction[-1] > extractor.distraction[0]


def test_extract_distraction_task_relevant_object_stays_at_zero(extractor):
    extractor.age = [30]
    extractor.driving_exp = ["more than 10 years"]
    # ROAD (id 9) is task-relevant (in the negative_indices set), so sustained gaze
    # on it should not accumulate positive distraction.
    extractor.gazed_objects = np.array([9] * 20, dtype=float)

    extractor.extract_distraction(smooth_win=1)

    assert len(extractor.distraction) == 20
    # Negative (task-relevant) contributions are floored to 0 by the final clip.
    assert np.all(extractor.distraction == 0.0)

def test_extract_hr(extractor):
    extractor.ppg = pd.Series(np.sin(np.linspace(0, 10*np.pi, 1000)))
    extractor.time_ms = np.arange(1000)
    x, locs = extractor.extract_hr()
    assert len(x) == 1000
    assert len(locs) > 0
    assert hasattr(extractor, 'RR')
    assert hasattr(extractor, 'hr')