import numpy as np
import time
import sys

from typing import List

sys.setrecursionlimit(2500)  # Increase recursion depth


# Dynamic programming algorithm 1 for the knapsack problem (NP complete). Time complexity = O(nw) (exponential time).
# Weights assumed to be integers.
# Space consumption can be further optimized if only final total value is to be computed.
# Code can be further vectorized for speed.
def knapsack_dp_1(vals: list, wgts: List[int], w: int, n: int) -> int:
    ans = np.zeros((n + 1, w + 1), dtype=np.int_)
    for idx in range(1, n + 1, 1):
        for x in range(w + 1):
            wgt = wgts[idx - 1]
            if x < wgt:
                ans[idx, x] = ans[idx - 1, x]
            else:
                ans[idx, x] = max(ans[idx - 1, x], ans[idx - 1, x - wgts[idx - 1]] + vals[idx - 1])
    return ans[n, w]


# Recursive version of the algorithm above.
# Only the answers needed are calculated, so saves time and space when w >> n.
def knapsack_dp_1_recursive(vals: list or np.ndarray, wgts: List[int] or np.ndarray, w: int, n: int, ans: dict) -> int:
    key_1 = str(n - 1) + ',' + str(w)
    if key_1 not in ans:
        ans[key_1] = knapsack_dp_1_recursive(vals[:-1], wgts[:-1], w, n - 1, ans)

    if w - wgts[-1] < 0:
        return ans[key_1]
    else:
        key_2 = str(n - 1) + ',' + str(w - wgts[-1])
        if key_2 not in ans:
            ans[key_2] = knapsack_dp_1_recursive(vals[:-1], wgts[:-1], w - wgts[-1], n - 1, ans)
        return max(ans[key_1], ans[key_2] + vals[-1])


# 2-step Greedy heuristic taking time O(nlogn).
# Accuracy guarantee: if max_weight < frac*knapsack capacity, then total_value > (1 - frac)*optimal_total_value
def knapsack_greedy_heuristic(vals: np.ndarray, wgts: np.ndarray, w: int, n: int) -> int:
    new_order = np.flip(np.argsort(vals / weights, kind='stable'))
    vals = vals[new_order]
    wgts = wgts[new_order]
    total_weight = 0
    total_values = 0
    for item_idx in range(n):
        if total_weight >= w:
            break
        if total_weight + wgts[item_idx] > w:
            continue
        else:
            total_weight += wgts[item_idx]
            total_values += vals[item_idx]
    return total_values


# Dynamic programming algorithm 2 for the knapsack problem (NP complete).
# Time/space complexity = O(n^2*max_value) (exponential).
def knapsack_dp_2(vals: List[int] or np.ndarray, wgts: list or np.ndarray, w: int, n: int) -> (int, List[int]):
    max_tot_value = np.sum(vals)
    records = np.empty((n + 1, max_tot_value + 1), dtype=np.bool_)
    current_ans = np.full(max_tot_value + 1, fill_value=np.inf)
    current_ans[0] = 0
    for idx in range(1, n + 1, 1):
        threshold = vals[idx - 1]
        temp = np.full_like(current_ans, fill_value=wgts[idx - 1])
        temp[threshold:] += current_ans[:max_tot_value + 1 - threshold]
        records[idx] = temp < current_ans
        current_ans = np.minimum(current_ans, temp)

    # The following line is inserted only to suppress automatic code inspection error
    largest_x = np.inf
    items_in_knapsack = []  # Number 1 in this list means the first item is in the knapsack
    for idx_x, min_tot_size in enumerate(np.flip(current_ans)):
        if min_tot_size <= w:
            largest_x = max_tot_value - idx_x  # The largest x for which A[n, x] <= W is to be returned
            # Track back for list of items added to the knapsack,time complexity = O(n)
            current_x = largest_x
            for idx in range(n, 0, -1):
                if not records[idx, current_x]:
                    continue
                else:
                    items_in_knapsack.append(idx)  # Item idx in knapsack
                    if current_x >= vals[idx - 1]:
                        current_x -= vals[idx - 1]
                    else:
                        break  # Terminate trace back once current_x drops below 0.
            break
    return largest_x, items_in_knapsack


# Heuristic version of dynamic programming algorithm 2. Time complexity = O(n^3/epsilon).
# Accuracy guarantee: for arbitrary epsilon, algo determines m s.t. total_value > (1 - epsilon)*optimal_total_value.
def knapsack_dp_2_heuristic(vals: List[int] or np.ndarray, wgts: list or np.ndarray, w: int, n: int,
                            epsilon: float) -> (int, List[int]):
    m = epsilon*np.max(vals)/n
    vals = np.array(vals, dtype=np.dtype('i4'))
    vals_rounded_down = np.floor(vals/m).astype(dtype=np.dtype('i4'))
    print(f"DP heuristic: m = {m}, max rounded down value = {max(vals_rounded_down)}")
    _, item_indices = knapsack_dp_2(vals_rounded_down, wgts, w, n)
    return np.sum(vals[np.array(item_indices, dtype=np.dtype('i2')) - 1]), item_indices  # Return total value of items


if __name__ == '__main__':
    # Test data
    # knapsack_size = 6
    # n_items = 4
    # values = [3, 2, 4, 4]
    # weights = [4, 3, 2, 3]

    # Part 1 real data
    print("Part 1:")
    values = []
    weights = []
    with open("AssignmentData/Data_knapsack1.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                knapsack_size, n_items = line.strip().split(' ')
                knapsack_size = int(knapsack_size)
                n_items = int(n_items)
            else:
                value, weight = line.strip().split(' ')
                values.append(int(value))
                weights.append(int(weight))

    start = time.time()
    result_1_dp_1 = knapsack_dp_1(values, weights, knapsack_size, n_items)
    print(f"Dynamic programming algo 1: result = {result_1_dp_1}, time consumed = {round(time.time() - start, 2)}s")
    start = time.time()
    result_1_dp_2, items = knapsack_dp_2(values, weights, knapsack_size, n_items)
    print(f"Dynamic programming algo 2: result = {result_1_dp_2}, time consumed = {round(time.time() - start, 2)}s")
    tot_value = np.sum(np.array(values)[np.array(items) - 1])
    assert tot_value == result_1_dp_1 == result_1_dp_2, "Results do not agree"
    print(f"Items in knapsack: {items}")

    # Part 1 dynamic programming heuristic
    start = time.time()
    result_1_dp_heuristic, items = knapsack_dp_2_heuristic(values, weights, knapsack_size, n_items, epsilon=0.1)
    print(f"DP heuristic: result = {result_1_dp_heuristic} = {result_1_dp_heuristic / result_1_dp_1}*optimal solution, "
          f"time consumed = {round(time.time() - start, 2)}s")
    print(f"Items in knapsack: {items}")

    # Part 2 real data
    print("\nPart 2:")
    values = []
    weights = []
    with open("AssignmentData/Data_knapsack_big.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                knapsack_size, n_items = line.strip().split(' ')
                knapsack_size = int(knapsack_size)
                n_items = int(n_items)
            else:
                value, weight = line.strip().split(' ')
                values.append(int(value))
                weights.append(int(weight))

    # Use np arrays to pass slice of array by reference -> to save memory.
    values = np.array(values, dtype=np.dtype('i4'))
    weights = np.array(weights, dtype=np.dtype('i4'))
    answers = dict()  # Use hash map to ensure O(1) insert and lookup of computed answers.
    for xx in range(knapsack_size + 1):
        answers['0' + ',' + str(xx)] = 0
    start = time.time()
    result_2_dp_1 = knapsack_dp_1_recursive(values, weights, knapsack_size, n_items, answers)
    print(f"Dynamic programming algo 1 (recursive): result = {result_2_dp_1}, "
          f"time consumed = {round(time.time() - start, 2)}s, ", end="")
    print(f"only {len(answers)} or {len(answers) / (n_items + 1) / (knapsack_size + 1)} "
          f"of all answer are needed and computed")

    # Part 2 greedy heuristic
    start = time.time()
    result_2_greedy = knapsack_greedy_heuristic(values, weights, knapsack_size, n_items)
    print(f"Greedy heuristic: result = {result_2_greedy} = {result_2_greedy / result_2_dp_1}*optimal solution, "
          f"time consumed = {round(time.time() - start, 3)}s")

    # Part 2 dynamic programming heuristic
    start = time.time()
    result_2_dp_heuristic, items = knapsack_dp_2_heuristic(values, weights, knapsack_size, n_items, epsilon=0.5)
    print(f"DP heuristic: result = {result_2_dp_heuristic} = {result_2_dp_heuristic / result_2_dp_1}*optimal solution, "
          f"time consumed = {round(time.time() - start, 2)}s")
    print(f"Items in knapsack: {items}")
