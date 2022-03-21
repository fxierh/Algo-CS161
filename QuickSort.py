import random
import time
import numpy as np


# Quick sort (unstable) in place, average case O(nlogn), worst case O(n^2) time complexity.
# O(logn) extra (not counting input) memory complexity for the call stack, since tail recursion is used.
def quick_sort(arr: list, leftmost_idx: int, rightmost_idx: int):
    if leftmost_idx == rightmost_idx:  # Base case: only 1 element to sort
        return
    else:
        # Choose pivot and swap it with the leftmost element (in the range to sort)
        pivot_idx = random.randint(leftmost_idx, rightmost_idx)
        pivot = arr[pivot_idx]
        arr[leftmost_idx], arr[pivot_idx] = arr[pivot_idx], arr[leftmost_idx]
        # Partitioning
        i = leftmost_idx + 1  # Leftmost element that is larger than pivot
        for j in range(leftmost_idx + 1, rightmost_idx + 1):
            if arr[j] < pivot:
                if j > i:  # Swap indices i and j only if element larger than pivot encountered at least once
                    arr[i], arr[j] = arr[j], arr[i]
                i += 1
        # Swap pivot to the correct position
        arr[leftmost_idx], arr[i - 1] = arr[i - 1], arr[leftmost_idx]
        # Recursions
        if i > leftmost_idx + 2:
            quick_sort(arr, leftmost_idx, i - 2)  # Subarray of elements smaller than pivot
        if i < rightmost_idx:
            quick_sort(arr, i, rightmost_idx)  # Subarray of elements larger than pivot


# Quick sort written for programming assignment #3
def quick_sort_count_comparison(arr: list, leftmost_idx: int, rightmost_idx: int, pivot_choice: str = "uniform"):
    global comparison_count
    if leftmost_idx == rightmost_idx:  # Base case: only 1 element to sort
        return 0
    else:
        # Choose pivot and swap it with the leftmost element (in the range to sort)
        comparison_count += rightmost_idx - leftmost_idx
        pivot_choices_implemented = ["uniform", "first", "last", "median_of_3"]
        assert pivot_choice in pivot_choices_implemented, f"Pivot choice should be one of {pivot_choices_implemented}."
        if pivot_choice == "uniform":
            pivot_idx = random.randint(leftmost_idx, rightmost_idx)
        elif pivot_choice == "first":
            pivot_idx = leftmost_idx
        elif pivot_choice == "last":
            pivot_idx = rightmost_idx
        else:  # pivot_choice == "median_of_3"
            pivot_indices = [leftmost_idx, (rightmost_idx - leftmost_idx)//2 + leftmost_idx, rightmost_idx]
            pivot_choices = np.array([arr[pivot_indices[0]], arr[pivot_indices[1]], arr[pivot_indices[2]]])
            pivot_idx = pivot_indices[np.argsort(pivot_choices)[1]]
        pivot = arr[pivot_idx]
        arr[leftmost_idx], arr[pivot_idx] = arr[pivot_idx], arr[leftmost_idx]
        # Partitioning
        i = leftmost_idx + 1  # Leftmost element that is larger than pivot
        for j in range(leftmost_idx + 1, rightmost_idx + 1):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        # Swap pivot to the correct position
        arr[leftmost_idx], arr[i - 1] = arr[i - 1], arr[leftmost_idx]
        # Recursions
        if i > leftmost_idx + 2:
            # Subarray of elements smaller than pivot
            quick_sort_count_comparison(arr, leftmost_idx, i - 2, pivot_choice)
        if i < rightmost_idx:
            # Subarray of elements larger than pivot
            quick_sort_count_comparison(arr, i, rightmost_idx, pivot_choice)


if __name__ == "__main__":
    # Generate random list (duplicate elements possible)
    len_arr = int(1e5)
    test = random.choices(range(2*len_arr), k=len_arr)
    # print(f"Initial array = {test}")
    correct_result = sorted(test)
    quick_sort(test, leftmost_idx=0, rightmost_idx=len_arr - 1)
    print(f"Sorted array = {test}")
    print(f"Correct result: {test == correct_result}")

    # Calculate average running time (inplace algorithm -> cannot use timeit module for timing)
    repetition = int(1e2)
    random.shuffle(test)
    total_duration = 0
    for iteration in range(repetition):
        start = time.time()
        quick_sort(test, leftmost_idx=0, rightmost_idx=len_arr - 1)
        total_duration += time.time() - start
        random.shuffle(test)
    # print(f"Total running time = {total_duration}s")
    print(f"Average running time = {total_duration/repetition}s")

    # Programming Assignment #3
    file1 = open("AssignmentData/Data_quick_sort.txt", 'r')
    lines = [int(line.strip()) for line in file1.readlines()]
    # print(f"\nProgramming Assignment #3: input array = {lines}")
    comparison_count = 0
    start = time.time()
    quick_sort_count_comparison(lines, leftmost_idx=0, rightmost_idx=len(lines) - 1, pivot_choice="median_of_3")
    print(f"Programming Assignment #3: quick sort time = {time.time() - start}s")
    print(f"Programming Assignment #3: correct result: {lines == sorted(lines)}")
    print(f"Programming Assignment #3: quick sort comparison count = {comparison_count}")
