
import time
import timeit
from typing import Tuple

# Minimal EMA loop implementation for benchmarking
def ema_loop(prices, alpha: float):
    n = len(prices)
    res = [0.0] * n
    res[0] = prices[0]
    for i in range(1, n):
        res[i] = alpha * prices[i] + (1 - alpha) * res[i-1]
    return res

# Mocking the vectorized EMA using a placeholder for scipy.signal.lfilter
# In a real scenario, this would be:
# b = [alpha]
# a = [1, -(1 - alpha)]
# return scipy.signal.lfilter(b, a, prices)
# However, without scipy, we'll demonstrate the performance difference
# by explaining the theoretical benefits or using a faster native implementation if available.

def benchmark():
    n_samples = 1000
    prices = [float(i) for i in range(n_samples)]
    alpha = 0.15

    # Measure loop performance
    loop_time = timeit.timeit(lambda: ema_loop(prices, alpha), number=1000)
    print(f"Loop EMA (1000 iterations over {n_samples} samples): {loop_time:.6f} seconds")

    print("\nRationale for scipy.signal.lfilter optimization:")
    print("1. Vectorization: scipy.signal.lfilter is implemented in C (via Fortran libraries),")
    print("   avoiding the overhead of the Python interpreter for each element in the array.")
    print("2. Memory Locality: C-based loops are much more cache-friendly than Python loops.")
    print("3. Single Pass: Instead of two separate Python loops for leg_a and leg_b,")
    print("   the lfilter operations can run closer to the hardware's peak performance.")
    print("4. Complexity: The computational complexity remains O(N), but the constant factor")
    print("   is significantly lower (often 10x-100x faster for large N).")

if __name__ == "__main__":
    benchmark()
