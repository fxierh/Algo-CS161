import numpy as np
import time


# Naive quadratic time complexity implementation
def brute_force_closest_pair(pts: np.ndarray) -> (np.ndarray, float):
    n_pts = pts.shape[0]

    closest_pts = np.array([None])
    closest_dist = np.inf
    for i in range(n_pts - 1):
        for j in range(i + 1, n_pts):
            pt1 = pts[i]
            pt2 = pts[j]
            d = np.linalg.norm(pt1 - pt2)
            if d < closest_dist:
                closest_dist = d
                closest_pts = np.vstack((pt1, pt2))
    return closest_pts, closest_dist


# Preprocessing step for the implementation of O(nlogn) time complexity
def preprocessing(pts: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    preprocessing: sort pts according to x/y coordinates
    :param pts: N*2 numpy array, first column x coordinates, second column y coordinates.
    """
    assert pts.shape[0] >= 2 and pts.shape[1] == 2
    pts_sorted_by_x = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    pts_sorted_by_y = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    return pts_sorted_by_x, pts_sorted_by_y


# O(nlogn) time complexity
def closest_pair(pts_sorted_by_x: np.ndarray, pts_sorted_by_y: np.ndarray) -> (np.ndarray, float):
    assert pts_sorted_by_x.shape == pts_sorted_by_y.shape, 'The 2 input arrays should have the same shape.'

    n_pts = pts_sorted_by_x.shape[0]
    if n_pts <= 2:  # Base case (<= 16 for better performance)
        closest_pts = np.array([None])
        closest_dist = np.inf
        for i in range(n_pts-1):
            for j in range(i+1, n_pts):
                pt1 = pts_sorted_by_x[i]
                pt2 = pts_sorted_by_x[j]
                d = np.linalg.norm(pt1 - pt2)
                if d < closest_dist:
                    closest_dist = d
                    closest_pts = np.vstack((pt1, pt2))
        return closest_pts, closest_dist
    else:
        # Split points into left and right half
        n_pts_left = n_pts // 2
        pts_sorted_by_x_left = pts_sorted_by_x[:n_pts_left]
        pts_sorted_by_x_right = pts_sorted_by_x[n_pts_left:]
        largest_x_left = pts_sorted_by_x_left[-1, 0]
        pts_sorted_by_y_left = pts_sorted_by_y[pts_sorted_by_y[:, 0] <= largest_x_left]  # Linear time complexity
        pts_sorted_by_y_right = pts_sorted_by_y[pts_sorted_by_y[:, 0] > largest_x_left]  # Linear time complexity

        # Recursive calls
        closest_pts_left, closest_dist_left = closest_pair(pts_sorted_by_x_left, pts_sorted_by_y_left)
        closest_pts_right, closest_dist_right = closest_pair(pts_sorted_by_x_right, pts_sorted_by_y_right)

        # Find the closest split point
        closest_dist_array = np.array([closest_dist_left, closest_dist_right])
        idx = np.argmin(closest_dist_array)
        delta = closest_dist_array[idx]

        # Filtering step: linear time complexity
        pts_sorted_by_y_strip = pts_sorted_by_y[np.logical_and(pts_sorted_by_y[:, 0] >= largest_x_left - delta,
                                                               pts_sorted_by_y[:, 0] <= largest_x_left + delta)]
        n_pts_strip = pts_sorted_by_y_strip.shape[0]
        closest_dist_split = delta
        closest_pts_split = 0
        for i in range(0, n_pts_strip):
            for j in range(1, 6):
                if i + j >= n_pts_strip:
                    continue
                else:
                    pt1 = pts_sorted_by_y_strip[i]
                    pt2 = pts_sorted_by_y_strip[i + j]
                    d = np.linalg.norm(pt1 - pt2)
                    if d < closest_dist_split:
                        closest_dist_split = d
                        closest_pts_split = np.vstack((pt1, pt2))

        # Combine results
        if closest_dist_split < delta:
            return closest_pts_split, closest_dist_split
        else:
            return closest_pts_left if idx == 0 else closest_pts_right, delta


if __name__ == "__main__":
    nb_points = int(1e3)
    test = np.random.randint(1, 100) * np.random.rand(nb_points, 2)
    points_sorted_by_x, points_sorted_by_y = preprocessing(test)
    start = time.time()
    closest_points, distance = closest_pair(points_sorted_by_x, points_sorted_by_y)
    print(f"Time = {time.time() - start}s")
    print(f"Closest points = {closest_points}\nDistance = {distance}")
    print(f"Consistent result: {distance == np.linalg.norm(closest_points[1] - closest_points[0])}")
    closest_points_ref, distance_ref = brute_force_closest_pair(test)
    print(f'''Correct result: {set(tuple(i.tolist()) for i in closest_points) 
                               == set(tuple(i.tolist()) for i in closest_points_ref) 
                               and distance == distance_ref}''')
