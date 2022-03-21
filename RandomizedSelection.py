import random
import numpy as np
import time


# Return the i-th smallest element of array, time complexity = O(n)
# Python list element access seems to be faster than numpy array element access
def randomized_selection(arr: list or np.ndarray, order_statistic: int, leftmost_idx: int, rightmost_idx: int) \
        -> int or float:
    if leftmost_idx == rightmost_idx:
        return arr[leftmost_idx]
    else:
        # Choose pivot and swap it with the leftmost element (in the range to sort)
        pivot_idx = random.randint(leftmost_idx, rightmost_idx)
        pivot = arr[pivot_idx]
        arr[leftmost_idx], arr[pivot_idx] = pivot, arr[leftmost_idx]
        # Partitioning
        i = leftmost_idx + 1  # Leftmost element that is larger than pivot
        for j in range(leftmost_idx + 1, rightmost_idx + 1):
            if arr[j] < pivot:
                if j > i:  # Swap indices i and j only if element larger than pivot encountered at least once
                    arr[i], arr[j] = arr[j], arr[i]
                i += 1
        # Swap pivot to the correct position
        pivot_idx = i - 1
        arr[leftmost_idx], arr[pivot_idx] = arr[pivot_idx], pivot
        if pivot_idx - leftmost_idx + 1 == order_statistic:
            return arr[pivot_idx]
        elif pivot_idx - leftmost_idx + 1 > order_statistic:
            return randomized_selection(arr, order_statistic, leftmost_idx, pivot_idx - 1)
        else:  # pivot_idx - leftmost_idx + 1 < order_statistic
            return randomized_selection(arr, order_statistic - (pivot_idx - leftmost_idx + 1), pivot_idx + 1,
                                        rightmost_idx)


# Another implementation, time complexity = O(n)
def randomized_selection_2(arr: list or np.ndarray, order_statistic: int) -> int or float:
    length_arr = len(arr)
    if length_arr == 1:  # Base case
        return arr[0]
    else:
        # Choose pivot and swap it with the leftmost element (in the range to sort)
        pivot_idx = random.randint(0, length_arr - 1)
        pivot = arr[pivot_idx]
        arr[0], arr[pivot_idx] = pivot, arr[0]
        # Partitioning
        i = 1  # Leftmost element that is larger than pivot
        for j in range(1, length_arr):
            if arr[j] < pivot:
                if j > i:  # Swap indices i and j only if element larger than pivot encountered at least once
                    arr[i], arr[j] = arr[j], arr[i]
                i += 1
        # Swap pivot to the correct position
        pivot_idx = i - 1
        arr[0], arr[pivot_idx] = arr[pivot_idx], pivot
        if pivot_idx + 1 == order_statistic:
            return arr[pivot_idx]
        elif pivot_idx + 1 > order_statistic:
            return randomized_selection_2(arr[:pivot_idx], order_statistic)
        else:  # pivot_idx - leftmost_idx + 1 < order_statistic
            return randomized_selection_2(arr[pivot_idx + 1:], order_statistic - pivot_idx - 1)


if __name__ == "__main__":
    # Generate random list (duplicate elements possible)
    len_arr = int(1e4)
    test = random.choices(range(2*len_arr), k=len_arr)
    order_stat = random.randint(1, len_arr)
    correct_result = sorted(test)[order_stat - 1]
    # result = randomized_selection(test, order_statistic=order_stat, leftmost_idx=0, rightmost_idx=len_arr - 1)
    result = randomized_selection_2(test, order_statistic=order_stat)
    print(f"Calculating order statistic {order_stat} of array")
    print(f"Result = {result}")
    print(f"Result is correct: {result == correct_result}")

    # Calculate average running time (inplace algorithm -> cannot use timeit module for timing)
    repetition = int(1e2)
    total_duration = 0
    for _ in range(repetition):
        random.shuffle(test)
        start = time.time()
        # randomized_selection(test, order_statistic=order_stat, leftmost_idx=0, rightmost_idx=len_arr - 1)
        randomized_selection_2(test, order_statistic=order_stat)
        total_duration += time.time() - start
    print(f"Total running time = {total_duration}s")
    print(f"Average running time = {total_duration/repetition}s")
