import random
import time
import numpy as np


# Check if a number is a power of 2 using bit manipulations
def pow_of_2_check(n: int) -> bool:
    assert n > 0, f"The number for power-of-2-check should be a POSITIVE integer. Here n = {n}."
    return n & (n - 1) == 0  # &: bit-wise and


# O(n**log_2^7) sub-cubic time complexity
def strassen_mat_mult(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    assert pow_of_2_check(m1.shape[0]) and m1.shape[0] == m1.shape[1] == m2.shape[0] == m2.shape[1], \
        f"Both mat should be of shape 2^n*2^n with the same n. Here the shapes are {m1.shape}, {m2.shape}."

    n = m1.shape[0]
    if n <= 2:  # Base case
        return m1 @ m2
    else:
        # Split m1 and m2 evenly into 4 sub-matrices each
        half_n = n//2
        a, b, c, d = m1[:half_n, :half_n], m1[:half_n, half_n:], m1[half_n:, :half_n], m1[half_n:, half_n:]
        e, f, g, h = m2[:half_n, :half_n], m2[:half_n, half_n:], m2[half_n:, :half_n], m2[half_n:, half_n:]
        # Compute the 7 products
        p1 = strassen_mat_mult(a, f-h)
        p2 = strassen_mat_mult(a+b, h)
        p3 = strassen_mat_mult(c+d, e)
        p4 = strassen_mat_mult(d, g-e)
        p5 = strassen_mat_mult(a+d, e+h)
        p6 = strassen_mat_mult(b-d, g+h)
        p7 = strassen_mat_mult(a-c, e+f)
        # Combine the results
        output_1 = p5 + p4 - p2 + p6
        output_2 = p1 + p2
        output_3 = p3 + p4
        output_4 = p1 + p5 - p3 - p7
    return np.concatenate((np.concatenate((output_1, output_2), axis=1), np.concatenate((output_3, output_4), axis=1)),
                          axis=0)


if __name__ == '__main__':
    # Test function "pow_of_2_check"
    test = 1483948
    print(f"The number is a power of 2: {pow_of_2_check(test)}")
    test = 2 ** 10
    print(f"The number is a power of 2: {pow_of_2_check(test)}")

    # Test Strassen matrix multiplication algorithm
    mat_size = 2**6
    mat_1 = random.randint(1, 100)*np.random.rand(mat_size, mat_size)
    mat_2 = random.randint(1, 100)*np.random.rand(mat_size, mat_size)
    ans = strassen_mat_mult(mat_1, mat_2)
    # print(f"Product = {ans}")
    print(f"Normalized Fresenius norm of (ans - np mat mult result): {np.linalg.norm(ans - mat_1@mat_2)/mat_size}")

    repetition = int(1e2)
    start = time.time()
    for _ in range(repetition):
        strassen_mat_mult(mat_1, mat_2)
    print(f"Strassen matrix multiplication average time = {(time.time() - start)/repetition}s")
