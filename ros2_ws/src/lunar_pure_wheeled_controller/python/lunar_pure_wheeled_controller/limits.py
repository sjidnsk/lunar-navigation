"""Joint speed/yaw slew limits for a fixed tracking curvature."""


def tracking_speed(
    curvature, desired_speed, measured_speed, measured_yaw_rate, dt, policy
):
    """Return feasible travel speed, or None when stopping is required.

    Intersect translational and angular acceleration intervals; independent
    clipping of omega would change the intended curvature.
    """
    lower = max(0.0, measured_speed - policy.max_linear_decel_mps2 * dt)
    upper = min(desired_speed, measured_speed + policy.max_linear_accel_mps2 * dt)
    yaw_lower = measured_yaw_rate - policy.max_angular_accel_radps2 * dt
    yaw_upper = measured_yaw_rate + policy.max_angular_accel_radps2 * dt
    if abs(curvature) <= 1e-9:
        if not yaw_lower <= 0.0 <= yaw_upper:
            return None
    else:
        a, b = sorted((yaw_lower / curvature, yaw_upper / curvature))
        lower, upper = max(lower, a), min(upper, b)
    return max(0.0, upper) if lower <= upper else None
