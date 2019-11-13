import jax.numpy as np
import jax
from time import time

start_time = time()

res_file = open('res.txt', 'w')

def f(x):
    return x[0] + x[1] * np.sin(x[2])

grad_f = jax.grad(f)

res = dict()

for i in range(1, 1000):
    # print("[", 10./i, ", ", 20./i, ", ", 30/i, "](f, grad_f): ",
    #     f(np.array([10./i, 20./i, 30./i])), grad_f(np.array([10./i, 20./i, 30./i])))
    res[str(f(np.array([10./i, 20./i, 30./i])))] = grad_f(np.array([10./i, 20./i, 30./i]))
    
    
    

end_time = time()
total_time = - start_time + end_time
print(f"\nFinished in {total_time} s...")

res_file.write("Running time: " + str(total_time) + " seconds...............\n\n\n\n")

for key in res:
    res_file.write(key)
    res_file.write(":\t\t")
    res_file.write(str(res[key]))
    res_file.write("\n")

res_file.close()


