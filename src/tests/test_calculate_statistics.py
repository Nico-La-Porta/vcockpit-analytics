import numpy as np
import pandas as pd
import pytest

from src.analysis.calculate_statistics import (
    compute_segment_slope,
    compute_overall_stats_per_scenario,
    compute_gazed_objects_per_scenario,
)


def test_compute_segment_slope_basic():
    values = np.array([0.0, 5.0, 10.0])
    time_values = np.array([0.0, 1.0, 2.0])
    # (last - first) / (last_time - first_time) = (10 - 0) / (2 - 0)
    assert compute_segment_slope(values, time_values) == 5.0


def test_compute_segment_slope_too_few_points_returns_nan():
    assert np.isnan(compute_segment_slope(np.array([1.0]), np.array([0.0])))
    assert np.isnan(compute_segment_slope(np.array([]), np.array([])))


def test_compute_segment_slope_ignores_nans():
    values = np.array([np.nan, 0.0, 10.0, np.nan])
    time_values = np.array([0.0, 1.0, 2.0, 3.0])
    # After dropping NaNs: values=[0, 10], time=[1, 2] -> slope = 10
    assert compute_segment_slope(values, time_values) == 10.0


def test_compute_overall_stats_per_scenario_averages_numeric_columns():
    df_stats = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3'],
        'scenario': ['City', 'City', 'Highway'],
        'mean_focus': [10.0, 20.0, 100.0],
    })
    result = compute_overall_stats_per_scenario(df_stats)

    assert set(result['scenario']) == {'City', 'Highway'}
    assert all(result['user_id'] == 'Total')

    city_row = result[result['scenario'] == 'City'].iloc[0]
    highway_row = result[result['scenario'] == 'Highway'].iloc[0]
    assert city_row['mean_focus'] == 15.0
    assert highway_row['mean_focus'] == 100.0
    # Column order should match the input dataframe.
    assert list(result.columns) == list(df_stats.columns)


def test_compute_gazed_objects_per_scenario_top_n():
    df = pd.DataFrame({
        'scenario': ['City', 'City'],
        'gazed_objects_names': [
            ['road', 'road', 'mirror'],
            ['road', 'dashboard', None],
        ],
        'segment_mask': [[1, 1, 2], [1, 2, 2]],
        'event_mask': [[False, False, True], [False, True, True]],
    })

    result = compute_gazed_objects_per_scenario(df, top_n_gazed=2, cut_segment7=False)

    assert len(result) == 1
    row = result.iloc[0]
    # Segment 1 ('NE1') collects gazed objects from both rows: ['road', 'road', 'road']
    # (mirror/dashboard fall into segment 2 = 'E1'), so 'road' is the sole top object.
    assert row['gazed_top2_NE1_obj'] == ['road']
    assert row['gazed_top2_NE1_perc'] == [100.0]
