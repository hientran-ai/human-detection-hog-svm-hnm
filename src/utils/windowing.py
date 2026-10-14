"""Sliding-window helpers."""


def sliding_window(image, step_size, window_size):
    """Yield ``(x, y, patch)`` windows, including image boundaries."""
    if step_size <= 0:
        raise ValueError("step_size must be greater than zero")

    window_width, window_height = window_size
    if window_width <= 0 or window_height <= 0:
        raise ValueError("window_size values must be greater than zero")

    image_height, image_width = image.shape[:2]
    if image_width < window_width or image_height < window_height:
        return

    max_x = image_width - window_width
    max_y = image_height - window_height
    x_positions = list(range(0, max_x + 1, step_size))
    y_positions = list(range(0, max_y + 1, step_size))

    if x_positions[-1] != max_x:
        x_positions.append(max_x)
    if y_positions[-1] != max_y:
        y_positions.append(max_y)

    for y in y_positions:
        for x in x_positions:
            yield x, y, image[y : y + window_height, x : x + window_width]
