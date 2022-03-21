from typing import List, Union
import random


# If multiple elements are equal to their indices, the following function only returns the index of one of them.
# Time complexity is O(log n)
def sorted_id_element_detection(sorted_arr: List[int], left_idx: int = 0) -> Union[int, None]:
    arr_len = len(sorted_arr)
    first_element = sorted_arr[0]
    if first_element > left_idx:  # Base case 1
        return None
    if first_element == left_idx:  # Base case 2
        return left_idx
    right_idx = left_idx + arr_len - 1
    last_element = sorted_arr[-1]
    if last_element < right_idx:  # Base case 3
        return None
    elif last_element == right_idx:  # Base case 4
        return left_idx + arr_len - 1
    else:  # Head element of sorted array < index while tail element of sorted array > index
        # Split sorted array in half
        left_len = arr_len // 2
        middle_idx = left_idx + left_len - 1  # Index of the last element of the left array
        if sorted_arr[left_len - 1] < middle_idx:
            return sorted_id_element_detection(sorted_arr[left_len:], middle_idx + 1)
        else:
            return sorted_id_element_detection(sorted_arr[:left_len], left_idx)


if __name__ == '__main__':
    array_length = random.randint(int(1e3), int(1e4))
    sorted_array = sorted(random.sample(range(-array_length, 2*array_length), array_length))
    correct_idx = None
    for i in range(array_length):
        if sorted_array[i] == i:
            correct_idx = i
            break

    idx = sorted_id_element_detection(sorted_array)
    print(f"Result = {idx}")
    print(f"Smallest correct result = {correct_idx}")
    print(f"Same result = {idx == correct_idx}")
    print(f"Correct result: {sorted_array[idx] == idx if idx else correct_idx is None}")

