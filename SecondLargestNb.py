import heapq  # For reference solution
import timeit

import numpy as np


# Number of comparisons = 3n/2 - 2
def second_largest_nb(arr: np.ndarray) -> list:
    # Returns [max element of arr, second-largest element of arr]
    len_arr = len(arr)
    if len_arr == 1:  # Base case 1
        return [arr[0], -np.inf]
    else:
        # Split array in half
        len_left = len_arr // 2
        left_arr, right_arr = arr[:len_left], arr[len_left:]
        max_left, second_largest_left = second_largest_nb(left_arr)
        max_right, second_largest_right = second_largest_nb(right_arr)
        if max_left > max_right:
            max_arr = max_left
            if max_right > second_largest_left:
                max_2_arr = max_right
            else:
                max_2_arr = second_largest_left
        else:
            max_arr = max_right
            if max_left > second_largest_right:
                max_2_arr = max_left
            else:
                max_2_arr = second_largest_right
        return [max_arr, max_2_arr]


# Find the largest array element recursively and store all elements that are directly compared
# with the largest element during calculation. Returns max element and runner-ups.
# Input is np array so that array slicing can be passed by reference.
# Number of comparisons = n - 1, number of runner-ups = log_2^n.
def find_largest(arr: np.ndarray) -> (int, list):
    if len(arr) == 1:  # Base case
        return arr[0], []
    else:
        # Split array in half
        len_left = len(arr) // 2
        left_arr, right_arr = arr[:len_left], arr[len_left:]
        max_left, runner_ups_left = find_largest(left_arr)
        max_right, runner_ups_right = find_largest(right_arr)
        if max_left > max_right:
            runner_ups_left.append(max_right)
            return max_left, runner_ups_left
        else:
            runner_ups_right.append(max_left)
            return max_right, runner_ups_right


# Better algorithm with time complexity = n + log_2^n - 2
def second_largest_nb_2(arr: np.ndarray) -> (int, int):
    max_ele, runner_ups = find_largest(arr)
    second_largest_ele = max(runner_ups)  # log_2^n - 1 comparisons needed here
    return max_ele, second_largest_ele


if __name__ == '__main__':
    test = np.arange(1e5, dtype=np.int_)
    np.random.shuffle(test)
    largest_ref, second_largest_ref = heapq.nlargest(2, test)
    print(f"Reference largest: {largest_ref}, reference second-largest: {second_largest_ref}")
    repetition = int(1e2)

    largest, second_largest = second_largest_nb(test)
    print(f"Implementation 1: largest: {largest}, second-largest: {second_largest}")
    print(f"Implementation 1 is correct: {largest == largest_ref and second_largest == second_largest_ref}")
    ave_time_1 = timeit.timeit(stmt='second_largest_nb(test)', number=repetition,
                               globals=globals()) / repetition
    print(f"Implementation 1: average running time = {ave_time_1}s")

    largest_2, second_largest_2 = second_largest_nb_2(test)
    print(f"Implementation 2: largest: {largest_2}, second-largest: {second_largest_2}")
    print(f"Implementation 2 is correct: {largest_2 == largest_ref and second_largest_2 == second_largest_ref}")
    ave_time_2 = timeit.timeit(stmt='second_largest_nb_2(test)', number=repetition,
                               globals=globals()) / repetition
    print(f"Implementation 2: average running time = {ave_time_2}s")
