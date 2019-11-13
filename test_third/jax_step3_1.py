import jax
import jax.numpy as np
import numpy as onp
import time

start = time.time()
NUM = 10**7
args = onp.zeros([NUM, 3])
for i in range(NUM):
    args[i, ] = [1000/NUM*i - 2, 100/NUM*i - 4, 10/NUM*i - 6]
args = np.asarray(args)
end = time.time()
print("Times to allocating memory: ", end - start, " s")
print(args)
# print(np.multiply(args[:,0], args[:,1]))

@jax.jit
def func(var):
    # (var[0]-args[:,0])**3
    temp = np.sin(np.power(np.subtract(var[0], args[:, 0]), 3))
    ele1 = np.sum(temp)
    # sin(var[1]**3 - var[2]**2) * args[:,1]
    temp = np.sin(np.subtract(np.power(var[1], 3), np.power(var[2], 2)))
    ele2 = np.sum(np.dot(temp, args[:, 1]))
    # log(args[:,1]*args[:,2]*var[0]*var[1]*var[2])
    temp = np.multiply(args[:, 1], args[:, 2])
    ele3 = np.sum(np.log(1 + np.abs(np.dot(var[0] * var[1] * var[2], temp))))
    # print("\n\n", ele1, "\n", ele2, "\n", ele3)
    # print(res, type(res))
    return ele1 + ele2 + ele3

grad = jax.grad(func)

print("\n\n\n")
start = time.time()
print("Gradient: ", grad(np.array([1., 1., 1.])))
end = time.time()
print("Times to calculate gradient: ", end - start, " s")

print("\n\n\n")
start = time.time()
print("Gradient: ", grad(np.array([2., 2., 2.])))
end = time.time()
print("Times to calculate gradient: ", end - start, " s")

print("\n\n\n")
start = time.time()
print("Gradient: ", grad(np.array([1.5, 1.5, 1.5])))
end = time.time()
print("Times to calculate gradient: ", end - start, " s")