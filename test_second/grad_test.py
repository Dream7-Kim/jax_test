import jax
import jax.numpy as np
import numpy as onp
import time

ARG_NUM = 3
TRY_NUM = 10000
STEP = 100/TRY_NUM

def f(x):
    return x[0]**2 + x[1] * np.sin(x[2])

# def sum_f(x):
#     ss = 0
#     for i in range(10000):
#         ss += x * np.log(f([i, i + 0.1, i + 0.3])**2)
#     return ss


res = onp.zeros([TRY_NUM, ARG_NUM])
for i in range(TRY_NUM):
    idx = STEP * i
    res[i] = onp.array([idx, idx + 1, idx + 2])

start = time.time()

grad_f = jax.grad(f)

print(grad_f(np.asarray(res)))
    

end = time.time()
print("Finished in ", end- start, " s... ... ...")