import time
from multiprocessing import Pool, cpu_count
from functools import partial


# Check if there exists two DISTINCT keys x and y in the hash table such that x + y = t. Time complexity = O(n).
def two_sum(t: int, d: dict) -> bool:
    for number in d.keys():
        if t - number in d:  # Time complexity of lookup = O(1).
            if number*2 != t:
                return True
    return False


if __name__ == '__main__':
    hash_table = dict()
    with open("AssignmentData/Data_2_Sum.txt", 'r') as f:
        for line in f:
            nb = int(line.strip())
            hash_table[nb] = 0  # Only key useful, value can be arbitrary (here 0)

    start = time.time()
    with Pool(cpu_count()) as p:
        results = p.map(partial(two_sum, d=hash_table), range(-10000, 10001))
    print(f"Multiprocessing solution: result = {sum(results)}, "
          f"time consumed = {round((time.time() - start)/60, 2)}min.")

    # start = time.time()
    # total = 0
    # for s in range(-10000, 10001):
    #     if two_sum(s, hash_table):
    #         total += 1
    # print(f"Naive solution: result = {total}, time consumed = {round((time.time() - start)/60, 2)}min.")
