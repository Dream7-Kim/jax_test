import jax
import jax.numpy as np
from time import time

start_time = time()

def f(x):
    return jax.numpy.asarray([x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jax.numpy.sin(x[0])])

print(jax.jacfwd(f)(np.array([1., 2., 3.])))


end_time = time()
total_time = - start_time + end_time
print(f"\nFinished in {total_time} s...")