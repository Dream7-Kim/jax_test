import jax.numpy as np
import jax
import numpy as onp
from time import time



def f(x):
    return x[0] + x[1] *np.sin(x[2])

grad_f = jax.grad(f)
res = onp.zeros([10000, 3])


start_time = time()

for i in range(1, 10000):
    iter_res = grad_f(np.array([1./i, 2./i, 3./i]))
    for j in range(3):
        res[i,j] = iter_res[j]


end_time = time()
total_time = - start_time + end_time
print(f"\nFinished in {total_time} s...")
print(onp.array2string(res))


# Write result to the file
res_file = open('res_test4.txt', 'w')
res_file.write("Running time: " + str(total_time) + " seconds...............\n\n\n\n")
for res_item in res:
    res_file.write(onp.array2string(res_item) + "\n")
res_file.close()


