import numpy as np
import pandas as pd
import pytest

from src.analysis.process_study import (
    create_event_and_segment_masks,
    count_lane_changes_per_segment_fixed,
    calculate_violation_percentage,
    count_distance_violation_changes_per_segment_fixed,
    filter_tasks,
    compute_fixation_durations,
    compute_tau_adjustment,
)


def test_create_event_and_segment_masks_single_event():
    df = pd.DataFrame({
        'vehicle_straight_driving_offset': [[0] * 10],
        'event_start_scenario': [[3]],
        'event_end_scenario': [[6]],
    })
    create_event_and_segment_masks(df)

    event_mask = df.loc[0, 'event_mask']
    segment_mask = df.loc[0, 'segment_mask']

    np.testing.assert_array_equal(
        event_mask, [False, False, False, True, True, True, False, False, False, False]
    )
    # Three contiguous blocks (NE, E, NE) -> three distinct segment ids, in order.
    assert list(segment_mask[:3]) == [segment_mask[0]] * 3
    assert list(segment_mask[3:6]) == [segment_mask[3]] * 3
    assert segment_mask[0] != segment_mask[3] != segment_mask[6]
    assert segment_mask[6] == segment_mask[9]


def test_create_event_and_segment_masks_no_events():
    df = pd.DataFrame({
        'vehicle_straight_driving_offset': [[0] * 5],
        'event_start_scenario': [[]],
        'event_end_scenario': [[]],
    })
    create_event_and_segment_masks(df)

    assert not any(df.loc[0, 'event_mask'])
    # No transitions -> a single segment id throughout.
    assert len(set(df.loc[0, 'segment_mask'])) == 1


def test_count_lane_changes_per_segment_fixed_shapes():
    lane = [1, 1, 2, 2, 2, 1, 1, 3]
    segment_mask = [1, 1, 2, 2, 2, 3, 3, 4]
    # inside (event) segments are capped at 3, outside (non-event) at 4.
    inside, outside = count_lane_changes_per_segment_fixed(lane, segment_mask, num_segments=4)
    assert len(inside) == 3
    assert len(outside) == 4


def test_count_lane_changes_per_segment_fixed_counts():
    # Lane changes happen at index 2 (1->2, segment 2) and index 5 (2->1, segment 3)
    # and index 7 (1->3, segment 4).
    lane = [1, 1, 2, 2, 2, 1, 1, 3]
    segment_mask = [1, 1, 2, 2, 2, 3, 3, 4]
    inside, outside = count_lane_changes_per_segment_fixed(lane, segment_mask, num_segments=4)
    # inside = even segments (2, 4) -> [seg2, seg4] = [1, 1]
    np.testing.assert_array_equal(inside, [1, 1, 0])
    # outside = odd segments (1, 3) -> [seg1, seg3] = [0, 1]
    np.testing.assert_array_equal(outside, [0, 1, 0, 0])


def test_calculate_violation_percentage_empty_returns_none():
    assert calculate_violation_percentage([], [True, False]) == (None, None)


def test_calculate_violation_percentage_basic():
    safety_distance_mask = [1, 1, 0, 0]
    event_mask = [True, True, False, False]
    inside_kept, outside_kept = calculate_violation_percentage(safety_distance_mask, event_mask)
    assert inside_kept == 100.0
    assert outside_kept == 0.0


def test_count_distance_violation_changes_per_segment_fixed_counts_1_to_0_transitions():
    # Violation changes 1->0 at index 2 (segment 1) and index 5 (segment 3).
    violation_mask = [1, 1, 0, 0, 1, 0, 1]
    segment_mask =   [1, 1, 1, 2, 3, 3, 4]
    inside, outside = count_distance_violation_changes_per_segment_fixed(violation_mask, segment_mask, num_segments=4)
    # outside = odd segments (1, 3) -> [1, 1]
    np.testing.assert_array_equal(outside, [1, 1, 0, 0])
    # inside = even segments (2, 4) -> [0, 0]
    np.testing.assert_array_equal(inside, [0, 0, 0])


def test_filter_tasks_keeps_only_acceptable_tasks():
    tasks = ['Call_0', 'ReachInitialSpeed', 'Call_1']
    starts = [0, 10, 20]
    ends = [5, 15, 25]
    start_subject = [1, 11, 21]
    execution = [4, 4, 4]
    reaction = [1, 1, 1]

    result = filter_tasks(['Call_0', 'Call_1'], tasks, starts, ends, start_subject, "City - calls", execution, reaction)
    filtered_tasks, filtered_starts, filtered_ends, filtered_start_subjects, filtered_execution, filtered_reaction = result

    assert filtered_tasks == ['Call_0', 'Call_1']
    assert filtered_starts == [0, 20]
    assert filtered_ends == [5, 25]


def test_filter_tasks_skips_filtering_for_tutorial():
    tasks = ['Anything']
    result = filter_tasks(['Call_0'], tasks, [0], [1], [0], "Tutorial", [1], [1])
    assert result == (tasks, [0], [1], [1], [1])


def test_compute_fixation_durations_resets_on_change():
    sequence = np.array([1, 1, 1, 2, 2, 1])
    durations = compute_fixation_durations(sequence)
    # First sample always 0 (loop starts at index 1); duration grows while value
    # stays the same and resets to 0 whenever it changes.
    np.testing.assert_array_equal(durations, [0, 1, 2, 0, 1, 0])


@pytest.mark.parametrize("age,exp,expected_sign", [
    (70, "more than 10 years", -1),   # old + experienced -> still net negative (age dominates)
    (30, "more than 10 years", 1),    # young-ish + experienced -> positive
    (40, "5-7 years", 0),             # neutral age bracket + neutral experience -> zero
])
def test_compute_tau_adjustment_sign(age, exp, expected_sign):
    adjustment = compute_tau_adjustment(age, exp)
    if expected_sign > 0:
        assert adjustment > 0
    elif expected_sign < 0:
        assert adjustment < 0
    else:
        assert adjustment == 0
