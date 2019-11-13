import jax
import jax.numpy as np
import time

# out = jax.pmap(lambda x: x ** 2)(np.arange(8))
# print(out)

print("Available devices: ", jax.devices())

x = np.arange(2 * 3 * 3.)
xx = x.reshape((2, 3, 3))
print("\n", xx)
y = np.arange(2 * 3 * 3.) ** 2
yy = y.reshape((2, 3, 3))
print("\n", yy , "\n")

# print("Result 1:")
# start = time.time()
# print(jax.pmap(np.dot)(xx, yy))
# end = time.time()
# print("Execution time with pmap: ", float(end-start))

print("Result 2:")
start = time.time()
print(np.dot(xx, yy))
end = time.time()
print("Execution time without pmap: ", float(end-start))
