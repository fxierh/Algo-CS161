import numpy as np
import math
import time
import gc


# Naive implementation of time complexity O(n*logn). Returns 1 or 2 weighted median(s) as a tuple.
def naive_weighted_median(elements: np.ndarray, weights: np.ndarray) -> tuple:
    temp = np.argsort(elements)
    # Sort elements and corresponding weights by first column, time complexity = O(nlogn)
    elements = elements[temp]
    weights = weights[temp]
    del temp
    gc.collect()
    weights = np.cumsum(weights)  # Time complexity = O(n)
    half_tot_weight = weights[-1] / 2
    weighted_medians = tuple()
    for idx, cumulative_weight in enumerate(weights):  # Time complexity = O(n)
        if cumulative_weight < half_tot_weight and \
                not math.isclose(cumulative_weight, half_tot_weight, rel_tol=0, abs_tol=1e-9):
            continue
        elif math.isclose(cumulative_weight, half_tot_weight, rel_tol=0, abs_tol=1e-9):
            weighted_medians += (elements[idx], )
        else:
            weighted_medians += (elements[idx], )
            break
    return weighted_medians


# Time complexity = O(n).
# arr: array of size n*2, 1st column contains elements, 2nd column contains corresponding weights.
def weighted_median(elements: np.ndarray, weights: np.ndarray, weight_threshold=0) -> tuple:
    nb_elements = len(elements)
    if nb_elements == 1:  # Base case
        return elements[0],
    else:
        # Partition array by median following the first column. Time complexity = O(n).
        median_idx = nb_elements // 2  # Index of median of the first column of arr.
        temp = np.argpartition(elements, median_idx)
        elements = elements[temp]
        weights = weights[temp]
        del temp
        gc.collect()

        # Split partitioned element/weight arrays in half. Time complexity = O(n).
        upper_weights = weights[:median_idx]
        sum_weights_upper = np.sum(upper_weights)
        if not weight_threshold:
            weight_threshold = np.sum(weights) / 2

        # Recursions
        if sum_weights_upper > weight_threshold and \
                not math.isclose(sum_weights_upper, weight_threshold, rel_tol=0, abs_tol=1e-9):
            return weighted_median(elements[:median_idx], upper_weights, weight_threshold)
        elif math.isclose(sum_weights_upper, weight_threshold, rel_tol=0, abs_tol=1e-9):
            return elements[median_idx - 1], elements[median_idx]
        else:
            lower_weights = weights[median_idx:]
            return weighted_median(elements[median_idx:], lower_weights, weight_threshold - sum_weights_upper.item())


if __name__ == "__main__":
    # test_elements = np.array([1, 2, 3, 4, 5])
    # np.random.shuffle(test_elements)
    # print(test_elements)
    # test_weights = np.array([0.1, 0.2, 0.2, 0.3, 0.2])
    correct_result = True
    iteration = 1
    while correct_result:
        print(f"Iteration {iteration} begins:")
        len_arr = int(1e7)
        np.random.seed(iteration)
        test_elements = np.random.random(len_arr)
        test_weights = 1e-10 + np.random.random(len_arr)  # Add small positive cst to avoid zero weight
        start = time.time()
        result = weighted_median(test_elements, test_weights)
        print(f"Linear time algorithm: {time.time() - start}s")
        start = time.time()
        ref_result = naive_weighted_median(test_elements, test_weights)
        correct_result = result == ref_result
        print(f"Naive algorithm: {time.time() - start}s")
        print(f"Result: {result}\nReference result: {ref_result}\nResult is correct: {correct_result}\n")
        iteration += 1
