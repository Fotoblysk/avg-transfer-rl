
import timeit

# Define the strings to be used in the comparison
substring = "hello"
string = "hello world"
exact_string = "hello world"

# Define the number of iterations for the timing
iterations = 1000000

# Time the "in" operation
in_time = timeit.timeit(stmt='"hello" in "hello world"', number=iterations)

# Time the "==" operation
eq_time = timeit.timeit(stmt='"hello world" == "hello world"', number=iterations)

print(f'Time for "hello" in "hello world": {in_time:.6f} seconds')
print(f'Time for "hello world" == "hello world": {eq_time:.6f} seconds')