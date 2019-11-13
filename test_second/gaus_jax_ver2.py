import jax
import jax.numpy as np
import numpy as onp
import time

NUM = 10**8
a = onp.zeros([NUM, ])
for i in range(NUM):
    a[i] = 10 / NUM * i - 2

arg = np.asarray(a)
print("****************** Prepared Arguments *******************")
print(arg)
print(type(arg))
print("*********************************************************\n\n\n\n")

def func(var = np.array([0.5, 1.])):
    temp = np.subtract(arg, var[1])
    temp = np.power(temp, 2)
    temp_sum = np.sum(temp)
    divider = np.power(var[0], 2)
    res = np.divide(temp_sum, divider)
    common = np.log(np.divide(1, var[0]))
    return np.dot(res, common)

grad = jax.grad(func)

start = time.time()
print(grad(np.array([0.5, 1])))
end = time.time()

print("Calculation time: ", end-start, " s ... ... ...")
