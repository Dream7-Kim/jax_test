import jax
import jax.numpy as np
import numpy as onp
import time


# np.power  np.add  np.subtract 

def sum_f(var):
    ret = 0
    for i in range(10**4):
        # x = 10 / 10**4 * i - 5
        x = np.subtract(np.divide(10, np.dot(np.power(10, 4), i)), 5)
        # ret -= (x - var[1])**2 / var[0]**2
        ret -= np.subtract(ret, np.divide(np.power(np.subtract(x, var[1]), 2), np.power(var[0], 2)))
    return np.dot(np.log(np.divide(1, var[0])), ret)

gradres = jax.grad(sum_f)

start = time.time()
print(gradres(np.array([1., 0.5])))
end = time.time()

print("Calculation time: ", end-start, " s ... ... ...")