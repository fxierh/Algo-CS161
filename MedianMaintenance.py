import heapq
import time


# Online median maintenance: give a sequence of numbers x1, ..., xn.
# At each step i, return the median of {x1,....,xi} with time complexity O(log(i)).
# h_low is max heap, h_high is min heap.
def median_maintenance(new_number: int) -> int:
    # Push new number into the correct heap, time complexity = O(log(i)).
    if h_low:
        if new_number < -h_low[0]:
            heapq.heappush(h_low, -new_number)
        else:
            heapq.heappush(h_high, new_number)
    else:
        heapq.heappush(h_low, -new_number)
    # Balance heap sizes if necessary, time complexity = O(1).
    if len(h_low) >= len(h_high) + 2:
        heapq.heappush(h_high, -heapq.heappop(h_low))
    elif len(h_high) >= len(h_low) + 2:
        heapq.heappush(h_low, -heapq.heappop(h_high))
    # Return median, time complexity = O(1).
    if len(h_low) < len(h_high):
        return h_high[0]
    else:
        return -h_low[0]


if __name__ == '__main__':
    f = open("AssignmentData/Data_median_maintenance.txt", 'r')
    lines = f.readlines()
    numbers = [int(nb.strip('\n')) for nb in lines]
    print(f"Number sequence: {numbers}")

    # h_low is max heap, h_high is min heap.
    h_low = []
    h_high = []
    medians = []
    start = time.time()
    for number in numbers:
        current_median = median_maintenance(number)
        medians.append(current_median)
    print(f"Time consumed: {round(time.time() - start, 2)}s.")
    print(f"The medians are {medians}.")
    print(f"The answer of programming assignment #3 is {sum(medians) % 10000}.")
