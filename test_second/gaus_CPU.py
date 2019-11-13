import autograd
import autograd.numpy as np
import numpy as onp
import time


num = 10**6

def sum_f(var):
    ret = 0
    for i in range(10**6):
        x = 10/10**6*i - 5
        ret += (-(x-var[1])**2/var[0]**2)
    return np.log(1/var[0])*ret

gradres = autograd.grad(sum_f)

start = time.time()
print(gradres([1., 0.5]))
end = time.time()

print("Calculation time: ", end-start, " s ... ... ...")