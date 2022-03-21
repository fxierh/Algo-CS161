import random
import heapq  # For reference solution
import numpy as np
import timeit


# Time complexity = 3n/2 - 2
def second_largest_nb(arr: list) -> list:
    # Returns a list of [max element of arr, second-largest element of arr]
    len_arr = len(arr)
    if len_arr == 1:  # Base case 1
        return [arr[0], -np.inf]
    # elif len_arr == 2:  # Base case 2
    #     return arr if arr[0] > arr[1] else arr[::-1]  # [::-1] is equivalent to array reverse
    else:
        # Split array in half
        len_left = len_arr // 2
        left_arr, right_arr = arr[:len_left], arr[len_left:]
        max_left, max_2_left = second_largest_nb(left_arr)
        max_right, max_2_right = second_largest_nb(right_arr)
        if max_left > max_right:
            max_arr = max_left
            if max_right > max_2_left:
                max_2_arr = max_right
            else:
                max_2_arr = max_2_left
        else:
            max_arr = max_right
            if max_left > max_2_right:
                max_2_arr = max_left
            else:
                max_2_arr = max_2_right
        return [max_arr, max_2_arr]


# Find the largest array element recursively and store all elements compared with the largest during calculation.
def find_largest(arr: list, runner_up: list):
    len_arr = len(arr)
    if len_arr == 1:  # Base case
        return arr[0]
    else:
        # Split array in half
        len_left = len_arr // 2
        left_arr, right_arr = arr[:len_left], arr[len_left:]
        max_left = find_largest(left_arr, runner_up)
        max_right = find_largest(right_arr, runner_up)
        if max_left > max_right:
            runner_up.append(max_right)
            return max_left
        else:
            runner_up.append(max_left)
            return max_right


# Better algorithm with time complexity = n + log_2^n - 2
def second_largest_nb_2(arr: list) -> list:
    runner_up = []
    max_arr = find_largest(arr, runner_up)
    max_2_arr = max(runner_up)
    return [max_arr, max_2_arr]


if __name__ == '__main__':
    random_array = [random.random() for _ in range(int(1e5))]
    largest_ref, second_largest_ref = heapq.nlargest(2, random_array)
    print(f"Reference largest: {largest_ref}, reference second-largest: {second_largest_ref}")
    repetition = int(1e2)

    largest, second_largest = second_largest_nb(random_array)
    print(f"Implementation 1: largest: {largest}, second-largest: {second_largest}")
    print(f"Implementation 1 is correct: {largest == largest_ref and second_largest == second_largest_ref}")
    ave_time_1 = timeit.timeit(stmt='second_largest_nb(random_array)', number=repetition,
                               globals=globals()) / repetition
    print(f"Implementation 1: average running time = {ave_time_1}s")

    largest_2, second_largest_2 = second_largest_nb_2(random_array)
    print(f"Implementation 2: largest: {largest_2}, second-largest: {second_largest_2}")
    print(f"Implementation 2 is correct: {largest_2 == largest_ref and second_largest_2 == second_largest_ref}")
    ave_time_2 = timeit.timeit(stmt='second_largest_nb_2(random_array)', number=repetition,
                               globals=globals()) / repetition
    print(f"Implementation 2: average running time = {ave_time_2}s")
