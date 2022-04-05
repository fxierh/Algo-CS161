import numpy as np
import time


# Linear time complexity. Return local min value, index (location) of local min.
# This implementation has a small issue, see
# https://stackoverflow.com/questions/18525179/find-local-minimum-in-n-x-n-matrix-in-on-time
def grid_local_min(grid: np.ndarray, offset: np.ndarray) -> (int or float, np.ndarray):
    (nb_row, nb_col) = grid.shape
    if nb_row < 3 or nb_col < 3:  # Base case
        argmin = np.unravel_index(grid.argmin(), grid.shape)
        return grid[argmin], np.array(argmin) + offset
    else:
        # Find minimum element of the concatenation of the middle row and the middle column
        middle_row_idx, middle_col_idx = nb_row // 2, nb_col // 2
        row_min_idx = np.argmin(grid[middle_row_idx, :])  # Min index of the middle row
        col_min_idx = np.argmin(grid[:, middle_col_idx])  # Min index of the middle column
        quadrant_idx = [0, 0]
        if grid[middle_row_idx, row_min_idx] < grid[col_min_idx, middle_col_idx]:  # Row min < column min
            min_element_idx = (middle_row_idx, row_min_idx)
            row_min = True
            if row_min_idx > middle_col_idx:
                quadrant_idx[1] = 1
        else:
            min_element_idx = (col_min_idx, middle_col_idx)
            row_min = False
            if col_min_idx > middle_row_idx:
                quadrant_idx[0] = 1
        min_element = grid[min_element_idx]

        # Get quadrant containing local min
        min_ele_nbrs = [grid[middle_row_idx - 1, row_min_idx], grid[middle_row_idx + 1, row_min_idx]] \
            if row_min else [grid[col_min_idx, middle_col_idx - 1], grid[col_min_idx, middle_col_idx + 1]]
        # If min element happens to be a local min
        if min_element_idx == (middle_row_idx, middle_col_idx) or min_element < min(min_ele_nbrs):
            return min_element, np.array(min_element_idx) + offset
        # Recursion
        else:
            if min_element > min_ele_nbrs[1]:
                if row_min:
                    quadrant_idx[0] = 1
                else:
                    quadrant_idx[1] = 1
            if quadrant_idx == [0, 0]:
                return grid_local_min(grid[:middle_row_idx, :middle_col_idx], offset=offset)
            elif quadrant_idx == [0, 1]:
                return grid_local_min(grid[:middle_row_idx, middle_col_idx + 1:],
                                      offset=np.array([offset[0], offset[1] + middle_col_idx + 1]))
            elif quadrant_idx == [1, 0]:
                return grid_local_min(grid[middle_row_idx + 1:, :middle_col_idx],
                                      offset=np.array([offset[0] + middle_row_idx + 1, offset[1]]))
            elif quadrant_idx == [1, 1]:
                return grid_local_min(grid[middle_row_idx + 1:, middle_col_idx + 1:],
                                      offset=np.array([offset[0] + middle_row_idx + 1, offset[1] + middle_col_idx + 1]))


if __name__ == '__main__':
    test_size = int(1e3)
    test = np.arange(test_size ** 2)
    # np.random.shuffle(test)
    test = test.reshape((test_size, test_size))
    test = np.flip(test, axis=0)
    print(f"Test 2D grid:\n{test}")

    start = time.time()
    result, [result_i, result_j] = grid_local_min(test, offset=np.array([0, 0]))
    print(f"Time = {time.time() - start}s")
    [ref_i], [ref_j] = np.where(test == result)
    result_neighbors = []
    if ref_i != 0:
        result_neighbors.append(test[ref_i - 1, ref_j])
    if ref_i != test_size - 1:
        result_neighbors.append(test[ref_i + 1, ref_j])
    if ref_j != 0:
        result_neighbors.append(test[ref_i, ref_j - 1])
    if ref_j != test_size - 1:
        result_neighbors.append(test[ref_i, ref_j + 1])
    print(f"Result = {result}, at location: {(result_i, result_j)}")
    print(f"Reference location = {ref_i, ref_j}, result is local min: {result < min(result_neighbors)}")
    print(f"Correct result: {result < min(result_neighbors) and result_i == ref_i and result_j == ref_j}")
