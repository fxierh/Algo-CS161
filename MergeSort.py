import time
import random


# O(nlogn) average/worst-case time complexity, stable sorting algorithm.
def merge_sort(arr: list) -> list:
    len_array = len(arr)

    if len_array <= 2:  # Base case
        return sorted(arr)
    else:
        # Split into two arrays and sort each recursively
        len_left = len_array // 2  # Length of the left array
        a = merge_sort(arr[:len_left])
        b = merge_sort(arr[len_left:])

        # Merge sorted arrays
        i = j = 0
        max_i, max_j = len_left - 1, len_array - len_left - 1
        output = []
        for _ in range(len_array):
            if i <= max_i and j <= max_j:
                if a[i] <= b[j]:  # Use "<=" instead of "<" to ensure the stability of sorting
                    output.append(a[i])
                    i += 1
                else:
                    output.append(b[j])
                    j += 1
            elif i > max_i:  # The "a" array runs out of elements
                output.extend(b[j:])
                break
            elif j > max_j:  # The "b" array runs out of elements
                output.extend(a[i:])
                break
        return output


if __name__ == "__main__":
    # Generate random list (possibly repeat elements)
    len_arr = int(1e5)
    test = random.choices(range(2*len_arr), k=len_arr)
    test_sorted = merge_sort(test)
    print(f"Initial array = {test}\nSorted array = {test_sorted}")
    print(f"Correct result: {merge_sort(test) == sorted(test)}")

    repetition = int(1e2)
    total_duration = 0
    for iteration in range(repetition):
        start = time.time()
        merge_sort(test)
        total_duration += time.time() - start
        random.shuffle(test)
    print(f"Merge sort average time = {total_duration/repetition}s")
