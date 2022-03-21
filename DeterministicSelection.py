import random
import time


# Return the i-th smallest element of array, time complexity = O(n) but slower than randomized selection in reality.
def deterministic_selection(arr: list, order_statistic: int) -> int or float:
    length_arr = len(arr)
    if length_arr == 1:  # Base case
        return arr[0]
    else:
        # Break array into groups of 5 then sort each group. The last group can have less than 5 elements.
        groups = [sorted(arr[i:i + 5]) for i in range(0, length_arr, 5)]
        # Extract median of each group
        medians = [group[len(group)//2] for group in groups]
        # Recursively compute the pivot as median of medians
        pivot = deterministic_selection(medians, order_statistic=len(medians)//2)
        # Get pivot index
        pivot_idx = arr.index(pivot)
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
            return deterministic_selection(arr[:pivot_idx], order_statistic)
        else:  # pivot_idx - leftmost_idx + 1 < order_statistic
            return deterministic_selection(arr[pivot_idx + 1:], order_statistic - pivot_idx - 1)


if __name__ == "__main__":
    # Generate random list (duplicate elements possible)
    len_arr = int(1e4)
    test = random.choices(range(2*len_arr), k=len_arr)
    order_stat = random.randint(1, len_arr)
    correct_result = sorted(test)[order_stat - 1]
    result = deterministic_selection(test, order_statistic=order_stat)
    print(f"Calculating order statistic {order_stat} of array")
    print(f"Result = {result}")
    print(f"Result is correct: {result == correct_result}")

    # Calculate average running time (inplace algorithm -> cannot use timeit module for timing)
    repetition = int(1e2)
    total_duration = 0
    for _ in range(repetition):
        random.shuffle(test)
        start = time.time()
        deterministic_selection(test, order_statistic=order_stat)
        total_duration += time.time() - start
    print(f"Total running time = {total_duration}s")
    print(f"Average running time = {total_duration/repetition}s")
