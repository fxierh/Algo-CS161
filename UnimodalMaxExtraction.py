from typing import Union  # For multiple possible output types
import random
import time


# Binary search, time complexity = O(logn)
def unimodal_max_extraction(unimodal_arr: list) -> Union[int, float]:
    # Python len() function returns an attribute that is already calculated. Thus, it runs O(1) time complexity.
    len_arr = len(unimodal_arr)
    if len_arr <= 2:  # Base case
        return max(unimodal_arr)
    else:
        len_left = len_arr // 2
        if unimodal_arr[len_left - 1] < unimodal_arr[len_left]:
            return unimodal_max_extraction(unimodal_arr[len_left:])
        else:
            return unimodal_max_extraction(unimodal_arr[:len_left])


if __name__ == '__main__':
    full_array_length = int(1e6)
    increasing_array_length = random.randint(1, full_array_length)
    unimodal_array = sorted([random.randint(1, 100)*random.random() for _ in range(increasing_array_length)])
    unimodal_array.extend(sorted([random.randint(1, 100)*random.random() for _ in
                                  range(full_array_length - increasing_array_length)], reverse=True))
    max_element_ref = max(unimodal_array)

    start = time.time()
    max_element = unimodal_max_extraction(unimodal_array)
    print(f"Time = {time.time() - start}s")
    print(f"Max element = {max_element}")
    print(f"Correct result: {max_element == max_element_ref}")
