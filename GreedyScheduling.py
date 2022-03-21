import numpy as np


# Greedy scheduling of O(nlogn) time complexity where n is the number of jobs.
def greedy_scheduling(weight_arr: np.ndarray, length_arr: np.ndarray, optimal: bool = False) -> int:
    if optimal:
        score_arr = weight_arr/length_arr
    else:
        score_arr = weight_arr - length_arr
    score_arr = np.array(list(zip(score_arr, weight_arr)), dtype=[('x', 'f8'), ('y', 'f8')])
    # O(nlogn) time complexity, other operations have linear time complexity.
    # Order attribute is for breaking ties. Same score -> job with larger weight first.
    job_seq = np.flip(np.argsort(score_arr, kind='stable', order=('x', 'y')))
    weight_arr = weight_arr[job_seq]
    finishing_time_arr = np.cumsum(length_arr[job_seq])
    return int(np.dot(weight_arr, finishing_time_arr).item())


if __name__ == '__main__':
    weight_array = np.zeros(10000)
    length_array = np.zeros(10000)
    with open("AssignmentData/Data_Scheduling.txt", 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                weight, length = [int(nb) for nb in line.strip().split(' ')]
                weight_array[i - 1] = weight
                length_array[i - 1] = length

    '''
    # Test data
    weight_array = np.array([3, 1, 4])
    length_array = np.array([5, 2, 6])
    '''

    weighted_completion_time = greedy_scheduling(weight_array, length_array, optimal=False)
    print(f"Answer  = {weighted_completion_time}")
    optimal_weighted_completion_time = greedy_scheduling(weight_array, length_array, optimal=True)
    print(f"Optimum = {optimal_weighted_completion_time}")
