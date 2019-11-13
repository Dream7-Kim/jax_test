import jax
import jax.numpy as np
import numpy as onp
import time


num = 10**6

def sum_f(var):
    ret = 0
    for i in range(10**4):
        x = 10 / 10**4 * i - 5
        ret -= (x - var[1])**2 / var[0]**2
    return np.log(1/var[0]) * ret

gradres = jax.grad(sum_f)

start = time.time()
print(gradres(np.array([1., 1.])))
end = time.time()

print("Calculation time: ", end-start, " s ... ... ...")