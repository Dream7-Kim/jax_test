import jax.numpy as np
from jax import jit
from timeit import default_timer as timer

def slow_f(x):
    return x * x + x * 2.0

x = np.ones((5000, 5000))
print(type(x))
fast_f = jit(slow_f)

start = timer()
slow_f(x)
end = timer()
print("Slow", end-start)

start = timer()
fast_f(x)
end = timer()
print("Fast(jit)", end-start)