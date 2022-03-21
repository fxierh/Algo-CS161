import random
import time


# Inplace stable sorting algorithm of worst/average-case complexity O(n^2) and best case complexity O(n).
def bubble_sort(array: list):
    len_array = len(array)

    # Needs to pass n - 1 elements at most
    for i in range(len_array - 1):
        swap = False
        for j in range(len_array - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swap = True

        # If array already sorted
        if not swap:
            break


if __name__ == "__main__":
    # Generate random list (possibly repeat elements)
    len_arr = int(1e4)
    test = random.choices(range(2 * len_arr), k=len_arr)
    test_init = test.copy()
    start = time.time()
    bubble_sort(test)
    print(f"Initial array = {test_init}\nSorted array = {test}")
    print(f"Correct result: {test == sorted(test_init)}, time consumed = {round(time.time() - start, 2)}s")

    # Calculate average running time (inplace algorithm -> cannot use timeit module for timing)
    repetition = int(1e1)
    random.shuffle(test)
    total_duration = 0
    for iteration in range(repetition):
        start = time.time()
        bubble_sort(test)
        total_duration += time.time() - start
        random.shuffle(test)
    print(f"Total running time = {total_duration}s")
    print(f"Average running time = {total_duration / repetition}s")
