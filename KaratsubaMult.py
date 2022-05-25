import math
import time


# Time complexity = O(n**log_2^3), where n is the number of digits of both inputs
def karatsuba_number_mult(x: int, y: int) -> int:
    assert x >= 0 and y >= 0, f"The two inputs should be NON-NEGATIVE integers. Here x = {x}, y = {y}."
    # Number of digits
    dig_x, dig_y = int(math.log10(x)) + 1 if x != 0 else 1, int(math.log10(y)) + 1 if y != 0 else 1

    if dig_x == 1 or dig_y == 1:  # Base case
        product = x*y
    else:
        # Split x, y into smaller numbers s.t. x = concatenate(ab), y = concatenate(cd). a, b, c, d are decimal numbers.
        min_dig = min(dig_x, dig_y)
        dig_bd = min_dig // 2
        dig_a = dig_x - dig_bd
        a = int(str(x)[:dig_a])  # First dig_x//2 digits of x
        b = x % 10 ** dig_bd  # The rest of x
        dig_c = dig_y - dig_bd
        c = int(str(y)[:dig_c])  # First dig_x//2 digits of y
        d = y % 10 ** dig_bd  # The rest of y

        ac = karatsuba_number_mult(a, c)
        bd = karatsuba_number_mult(b, d)
        product = 10 ** (2 * dig_bd) * ac + 10 ** dig_bd * (karatsuba_number_mult(a + b, c + d) - ac - bd) + bd
    return product


if __name__ == '__main__':
    num_1 = 3141592653589793238462643383279502884197169399375105820974944592
    num_2 = 2718281828459045235360287471352662497757247093699959574966967627
    ans = karatsuba_number_mult(num_1, num_2)
    print(f"Product = {ans}")
    print(f"Correct result: {ans == num_1 * num_2}")

    repetition = int(1e2)
    start = time.time()
    for i in range(repetition):
        karatsuba_number_mult(num_1, num_2)
    print(f"Karatsuba average time = {(time.time() - start)/repetition}s")
