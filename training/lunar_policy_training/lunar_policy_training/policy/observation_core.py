"""Observed-only hierarchical policy observation constants."""
SOURCE_OBSERVATION_SCHEMA_VERSION = 'policy_observation/v1'
GLOBAL_PRIOR_CHANNELS = ('height_prior', 'value_prior', 'obstacle_prior', 'traversability_prior', 'current_position_marker_lowres', 'heading_sin_marker_lowres', 'heading_cos_marker_lowres')
COVERAGE_SUMMARY_CHANNELS = ('observed_ratio', 'unknown_ratio', 'observed_free_ratio', 'observed_blocked_ratio', 'frontier_count', 'observed_safe_frontier_count', 'reachable_frontier_count', 'mean_uncovered_value_prior')
LOCAL_CROP_CHANNELS = ('observed_height', 'coverage_mask', 'obstacle', 'traversability', 'local_frontier_channel', 'current_position_marker', 'heading_sin_marker', 'heading_cos_marker')
FRONTIER_FEATURE_FIELDS = ('x_norm', 'y_norm', 'distance_from_robot_norm', 'bearing_sin', 'bearing_cos', 'potential_coverage_gain_norm', 'visible_unknown_count_norm', 'value_gain_norm', 'frontier_segment_id_norm', 'segment_length_norm', 'normal_sin', 'normal_cos', 'normal_confidence', 'candidate_generation_mode', 'recommended_theta_sin', 'recommended_theta_cos', 'traversability', 'clearance_norm', 'reachable_prefilter_cost_norm', 'region_coverage_ratio', 'region_unknown_ratio', 'same_connected_component')
POSE_FEATURE_FIELDS = ('x_norm', 'y_norm', 'sin(theta)', 'cos(theta)', 'observed_roi_ratio', 'remaining_step_budget_norm')
