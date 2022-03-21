import time
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import repeat


# Naive quadratic time complexity implementation
def get_inv_count(arr: list) -> int:
    n = len(arr)
    inversion_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversion_count += 1
    return inversion_count


# Naive quadratic time complexity implementation using multiprocessing
def get_inv_count_multiprocessing(i: int, arr: list) -> int:
    n = len(arr)
    inversion_count = 0
    for j in range(i + 1, n):
        if arr[i] > arr[j]:
            inversion_count += 1
    return inversion_count


# O(nlogn) time complexity. Adapted from merge sort.
def inversion_counter(arr: list) -> (list, int):
    len_arr = len(arr)

    if len_arr < 2:  # Base case 1
        return arr, 0
    elif len_arr == 2:  # Base case 2
        return [min(arr), max(arr)], 1 if arr[0] > arr[1] else 0
    else:
        # Split into two arrays, sort and count inversions for each one separately
        len_left = len_arr // 2  # Length of the left array
        a, left_inv_count = inversion_counter(arr[:len_left])
        b, right_inv_count = inversion_counter(arr[len_left:])

        # Merge sorted arrays while counting split inversions
        i = j = 0
        max_i, max_j = len_left - 1, len_arr - len_left - 1
        output = []
        split_inv_count = 0
        for _ in range(len_arr):
            if i <= max_i and j <= max_j:
                if a[i] <= b[j]:
                    output.append(a[i])
                    i += 1
                    split_inv_count += j  # j = # of right array elements already inserted into the output
                else:
                    output.append(b[j])
                    j += 1
            elif i > max_i:  # The "a" array runs out of elements
                output.extend(b[j:])
                break
            elif j > max_j:  # The "b" array runs out of elements
                output.extend(a[i:])
                split_inv_count += (max_i - i + 1)*j
                break
        return output, left_inv_count + right_inv_count + split_inv_count


if __name__ == '__main__':
    # Generate random list (possibly repeat elements)
    test = random.choices(range(1000), k=500)
    test_sorted, inv_count = inversion_counter(test)
    reference_inv_count = get_inv_count(test)
    print(f"Test array = {test}")
    print(f"Sorted array = {test_sorted}")
    print(f"Inversion count = {inv_count}")
    print(f"Correct result: {bool((inv_count == reference_inv_count)*(test_sorted == sorted(test)))}")

    repetition = int(1e2)
    start = time.time()
    for iteration in range(repetition):
        inversion_counter(test)
    print(f"Inversion counter average time = {(time.time() - start)/repetition}s")

    start = time.time()
    for iteration in range(repetition):
        get_inv_count(test)
    print(f"Naive inversion counter average time = {(time.time() - start)/repetition}s")

    # Programming Assignment #2
    file1 = open("AssignmentData/Data_inv_counter.txt", 'r')
    lines = [int(line.strip()) for line in file1.readlines()]
    start = time.time()
    _, inv_count = inversion_counter(lines)
    print(f"\nProgramming Assignment #2: inversion counter time = {time.time() - start}s")
    print(f"Programming Assignment #2: inversion count = {inv_count}")

    start = time.time()
    with Pool(cpu_count() // 2) as p:
        result = p.map(partial(get_inv_count_multiprocessing, arr=lines), range(len(lines)))
        # result = p.starmap(get_inv_count, zip(range(len(lines)), repeat(lines)))  # Equivalent to the previous line
    print(f"Programming Assignment #2: naive multiprocessing inversion counter time = {time.time() - start}s")
    print(f"Programming Assignment #2: multiprocessing inversion count = {sum(result)}")

    start = time.time()
    reference_inv_count = get_inv_count(lines)
    print(f"Programming Assignment #2: naive inversion counter time = {(time.time() - start)/60}min")
    print(f"Programming Assignment #2: reference inversion count = {reference_inv_count}")
