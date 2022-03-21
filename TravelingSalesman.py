import math
import time
import numpy as np
import gc

from typing import List, Dict
from KruskalMST import n_choose_k_comb


# Exact solution of the TSP problem using dynamic programming.
# Time complexity = O(n^2*2^n). Space complexity = O(n*2^n). Code optimized for memory.
def traveling_salesman(dist_mat: np.ndarray, pb_size: int) -> float:
    # A[S, i] = sub_pb_results[i][S] where sub_pb_results[i] is a dictionary
    # and S (with the first point excluded, i.e. starts from the 2nd point)
    # is a set in binary representation (to save memory).
    sub_pb_results: List[Dict[int]] = [{} for _ in range(pb_size)]
    sub_pb_results[0][0] = 0
    # Used to transform a set into binary representation
    powers_of_2 = [2 ** power for power in range(pb_size - 1)]

    for sub_pb_size in range(1, pb_size, 1):
        # Only save sub-problem results of the previous problem size, to save memory.
        new_sub_pb_results: List[Dict[int]] = [{} for _ in range(pb_size)]
        # Get all combinations of points of size sub_pb_size.
        # Each element comb of combs denotes a set S (with the first point excluded).
        combs = n_choose_k_comb(pb_size - 1, sub_pb_size)
        for comb in combs:
            offset_binary = sum([powers_of_2[pt] for pt in comb])
            for j in comb:  # j + 1 represents a point in S - {first point}
                # previous_sub_pb_key = S - {j} in binary
                previous_sub_pb_key = offset_binary - powers_of_2[j]
                # previous_results: A[S - {j}, k] + distance(k, j) for k in S - {j}
                previous_results = []
                for k in set([0] + [ele + 1 for ele in comb]) - {j + 1}:
                    if k == 0 and sub_pb_size != 1:
                        continue
                    previous_results.append(sub_pb_results[k][previous_sub_pb_key] + dist_mat[k, j + 1])
                new_sub_pb_results[j + 1][offset_binary] = min(previous_results)
        sub_pb_results = new_sub_pb_results
        print(f"Sub-problems of size {sub_pb_size} finished, "
              f"total number of dict entries = {sum([len(ele) for ele in sub_pb_results])}")
    return min([sub_pb_results[j][2 ** (pb_size - 1) - 1] + dist_mat[j, 0] for j in range(1, pb_size, 1)])


# Nearest neighbor heuristic of the TSP problem. Time/space complexity = O(n^2).
def traveling_salesman_nn_heuristic(dist_mat: np.ndarray, pb_size: int) -> (float, List[int]):
    visited_cities = [0]
    total_distance = 0.0
    last_city = 0
    back_to_0_distance = dist_mat[:, 0].copy()
    dist_mat[:, 0] = np.inf  # Mark city 0 as visited
    for _ in range(pb_size - 1):
        # np.argmin breaks ties by returning the smallest index
        current_city = np.argmin(dist_mat[last_city, :])
        total_distance += dist_mat[last_city, current_city]
        visited_cities.append(current_city)
        dist_mat[:, current_city] = np.inf  # Mark current city as visited
        last_city = current_city
    # Return to city 0
    total_distance += back_to_0_distance[last_city]
    return total_distance, visited_cities


if __name__ == '__main__':
    # Test data
    # n_pts = 4
    # distance_matrix = np.zeros((n_pts, n_pts))  # distance_matrix[i, j] = distance(pt_i, pt_j)
    # distance_matrix[0, 1] = distance_matrix[1, 0] = 2
    # distance_matrix[0, 2] = distance_matrix[2, 0] = 1
    # distance_matrix[0, 3] = distance_matrix[3, 0] = 4
    # distance_matrix[1, 2] = distance_matrix[2, 1] = 3
    # distance_matrix[1, 3] = distance_matrix[3, 1] = 5
    # distance_matrix[2, 3] = distance_matrix[3, 2] = 6

    # Real data for exact solution
    coords = []
    with open("AssignmentData/Data_tsp.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                x, y = line.strip().split(' ')
                x, y = float(x), float(y)
                coords.append((x, y))
    n_pts = len(coords)

    # Use 32-bit float to reduce memory consumption
    distance_matrix = np.zeros((n_pts, n_pts), dtype=np.dtype('f4'))  # distance_matrix[i, j] = distance(pt_i, pt_j)
    for idx1 in range(n_pts):
        x1, y1 = coords[idx1]
        for idx2 in range(idx1 + 1, n_pts, 1):
            x2, y2 = coords[idx2]
            distance_matrix[idx1, idx2] = distance_matrix[idx2, idx1] = math.hypot(x1 - x2, y1 - y2)

    start = time.time()
    result = traveling_salesman(distance_matrix, n_pts)
    print(f"Result = {result}, time consumed = {round(time.time() - start)}s")

    # Real data for heuristic
    print("Nearest neighbor heuristic:")
    x_array = []
    y_array = []
    with open("AssignmentData/Data_tsp_big.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                x, y = line.strip().split(' ')[1:]
                x, y = float(x), float(y)
                x_array.append(x)
                y_array.append(y)
    n_pts = len(x_array)
    # Cannot use smaller precision, solution will be incorrect!
    x_array = np.array(x_array, dtype=np.dtype('f8'))
    y_array = np.array(y_array, dtype=np.dtype('f8'))

    # Use 32-bit float to reduce memory consumption
    distance_matrix = np.empty((n_pts, n_pts), dtype=np.dtype('f8'))  # distance_matrix[i, j] = distance(pt_i, pt_j)
    for idx1 in range(n_pts):
        x1, y1 = x_array[idx1], y_array[idx1]
        distance_matrix[idx1, :] = np.hypot(x_array - x1, y_array - y1)

    del x_array, y_array
    gc.collect()

    start = time.time()
    result, sequence = traveling_salesman_nn_heuristic(distance_matrix, n_pts)
    print(f"Result = {result}, time consumed = {round(time.time() - start)}s")
    print(f"City visit sequence = {sequence}")
    assert len(set(sequence)) == n_pts
