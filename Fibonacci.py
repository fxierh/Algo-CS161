# Compute the first idx terms of the Fibonacci series in O(n) time and O(n) extra memory, using dynamic programming.
def fibonacci(idx: int) -> list:
    ans = [0]*idx
    ans[0] = 0
    ans[1] = 1
    if idx < 3:
        return ans[:idx]
    for i in range(2, idx, 1):
        ans[i] = ans[i - 1] + ans[i - 2]
    return ans[:idx]


if __name__ == '__main__':
    print(fibonacci(idx=100))
