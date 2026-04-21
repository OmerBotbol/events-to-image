import math


def calculate_object_velocity(
    screen_width_m: float,
    camera_distance_m: float,
    duration_s: float,
    camera_resolution_px: tuple[int, int],
    roi_x_start_px: int = None,
    roi_x_end_px: int = None,
) -> dict:
    """
    Calculate the velocity of an object moving across a screen filmed by a camera.

    Args:
        screen_width_m:        Physical width of the screen in meters.
        camera_distance_m:     Distance from the camera lens to the screen surface in meters.
        duration_s:            Time the object takes to travel from start to end position (seconds).
        camera_resolution_px:  Camera resolution as (width_px, height_px).
        roi_x_start_px:        Pixel x-coordinate in the camera frame where the movement starts.
                               Defaults to 0 (left edge of frame).
        roi_x_end_px:          Pixel x-coordinate in the camera frame where the movement ends.
                               Defaults to camera frame width (right edge).

    Returns:
        dict with:
            linear_velocity_m_s:    Physical velocity on the screen surface (m/s).
            angular_velocity_rad_s: Angular velocity as seen from the camera lens (rad/s).
            angular_velocity_deg_s: Same, in degrees/s.
            displacement_m:         Physical distance travelled on screen (m).
            total_angle_rad:        Total angle swept at the camera lens (rad).
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if camera_distance_m <= 0:
        raise ValueError("camera_distance_m must be positive")
    if screen_width_m <= 0:
        raise ValueError("screen_width_m must be positive")

    cam_width_px, _ = camera_resolution_px
    if cam_width_px <= 0:
        raise ValueError("camera resolution width must be positive")

    if roi_x_start_px is None:
        roi_x_start_px = 0
    if roi_x_end_px is None:
        roi_x_end_px = cam_width_px

    if not (0 <= roi_x_start_px <= cam_width_px and 0 <= roi_x_end_px <= cam_width_px):
        raise ValueError("ROI pixel coordinates must be within camera resolution")

    start_x_fraction = roi_x_start_px / cam_width_px
    end_x_fraction   = roi_x_end_px   / cam_width_px

    displacement_m = abs(end_x_fraction - start_x_fraction) * screen_width_m
    linear_velocity_m_s = displacement_m / duration_s

    # Angular positions relative to the camera's optical axis (centred on the screen)
    screen_centre_x = screen_width_m / 2.0
    x_start = (start_x_fraction * screen_width_m) - screen_centre_x
    x_end   = (end_x_fraction   * screen_width_m) - screen_centre_x

    angle_start_rad = math.atan2(x_start, camera_distance_m)
    angle_end_rad   = math.atan2(x_end,   camera_distance_m)

    total_angle_rad        = abs(angle_end_rad - angle_start_rad)
    angular_velocity_rad_s = total_angle_rad / duration_s
    angular_velocity_deg_s = math.degrees(angular_velocity_rad_s)

    return {
        "linear_velocity_m_s":    linear_velocity_m_s,
        "angular_velocity_rad_s": angular_velocity_rad_s,
        "angular_velocity_deg_s": angular_velocity_deg_s,
        "displacement_m":         displacement_m,
        "total_angle_rad":        total_angle_rad,
    }

