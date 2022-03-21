import random
import time
import numpy as np


# Time complexity = O(logn)
def binary_search(sorted_arr: np.ndarray, ele_to_search) -> int or None:
    len_arr = len(sorted_arr)
    if len_arr == 1:
        if sorted_arr[0] == ele_to_search:
            return 0
        else:
            return None
    else:
        # Split array in half
        len_left = len_arr//2
        left_arr, right_arr = sorted_arr[:len_left], sorted_arr[len_left:]

        if ele_to_search > left_arr[-1]:
            relative_idx = binary_search(right_arr, ele_to_search)
            if relative_idx is None:
                return None
            else:
                return len_left + relative_idx
        else:
            return binary_search(left_arr, ele_to_search)


if __name__ == "__main__":
    array_length = int(1e7)
    sorted_array = np.sort(np.round(np.random.rand(array_length), decimals=7))
    random_idx = random.randint(0, array_length)
    element_to_search = sorted_array[random_idx]
    print(f"Searching element {element_to_search}, index {random_idx} of the original sorted array.")

    start = time.time()
    idx = binary_search(sorted_array, element_to_search)
    print(f"Time = {time.time() - start}s")
    print(f"Result: {idx}")
    print(f"Correct result: {sorted_array[idx] == element_to_search if idx else element_to_search not in sorted_array}")
