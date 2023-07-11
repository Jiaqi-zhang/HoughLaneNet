import numpy as np
# import numba


###########################################################
## Adapted from Deep Hough Transform
## https://github.com/Hanqer/deep-hough-transform


# @numba.jit(nopython=True)
def angle_of_line(y0, x0, y1, x1):
    """
    Returns:
        range: [-pi/2, pi/2]
    """
    if x0 == x1:
        return -np.pi / 2
    return np.arctan((y0 - y1) / (x0 - x1))


# @numba.jit(nopython=True)
def line_to_hough(line, img_size):
    """
    Converts a line segment into hough space point.

    Args:
        line: (y0, x0, y1, x1) line coordinates.

    Returns:
        Hough space point (y, x)
    """
    H, W = img_size
    theta = angle_of_line(*line)
    alpha = theta + np.pi / 2  # [0, pi]
    y0, x0, _, _ = line
    if theta == -np.pi / 2:
        r = x0 - W / 2
    else:
        k = np.tan(theta)
        y1 = y0 - H / 2
        x1 = x0 - W / 2
        r = (y1 - k * x1) / np.sqrt(1 + k ** 2)
    return alpha, r


# @numba.jit(nopython=True)
def line_to_hough_space(line, num_angle, num_rho, img_size):
    """
    Converts a line segment into quantized hough space point.

    Args:
        line: (y0, x0, y1, x1) line coordinates.
        num_angle, num_rho: hough space size.
        img_size: image size.

    Returns:
        Quantized hough space point (y : int, x : int)
    """
    H, W = img_size
    theta, r = line_to_hough(line, img_size)

    irho = int(np.sqrt(H * H + W * W) + 1) / (num_rho - 1)
    itheta = np.pi / num_angle

    r = int(np.round(r / irho)) + int((num_rho) / 2)
    theta = int(np.round(theta / itheta))
    if theta >= num_angle:
        theta = num_angle - 1
    return theta, r


def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y, x with angle, return a two point in image boundary with shape [H, W]
    Returns:
        point:[x, y]
    '''
    point1 = None
    point2 = None

    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H - 1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W - 1, y)
    else:
        k = np.tan(angle)
        if y - k * x >= 0 and y - k * x < H:  # left
            if point1 == None:
                point1 = (0, int(y - k * x))
            elif point2 == None:
                point2 = (0, int(y - k * x))
                if point2 == point1: point2 = None

        if k * (W - 1) + y - k * x >= 0 and k * (W - 1) + y - k * x < H:  # right
            if point1 == None:
                point1 = (W - 1, int(k * (W - 1) + y - k * x))
            elif point2 == None:
                point2 = (W - 1, int(k * (W - 1) + y - k * x))
                if point2 == point1: point2 = None

        if x - y / k >= 0 and x - y / k < W:  # top
            if point1 == None:
                point1 = (int(x - y / k), 0)
            elif point2 == None:
                point2 = (int(x - y / k), 0)
                if point2 == point1: point2 = None

        if x - y / k + (H - 1) / k >= 0 and x - y / k + (H - 1) / k < W:  # bottom
            if point1 == None:
                point1 = (int(x - y / k + (H - 1) / k), H - 1)
            elif point2 == None:
                point2 = (int(x - y / k + (H - 1) / k), H - 1)
                if point2 == point1: point2 = None

        if point2 == None: point2 = point1
    return point1, point2


def hough_points_to_line(point_list, num_angle, num_rho, img_size):
    """
    Converts a quantized hough space point into line.

    Args:
        point_list: quantized hough space point list [(y : int, x : int)]
        num_angle, num_rho: hough space size.
        img_size: image size.

    Returns:
        line: list of (y0, x0, y1, x1) line coordinates.
    """
    # return type: [(y1, x1, y2, x2)]
    H, W = img_size
    irho = int(np.sqrt(H * H + W * W) + 1) / ((num_rho - 1))
    itheta = np.pi / num_angle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - num_rho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H - 1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(-cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points
