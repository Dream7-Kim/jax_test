import jax
from jax import jit
from jax import jacrev
import jax.numpy as np
import numpy as onp
import time

import autograd
import autograd.numpy as anp

NUM_VAR = 10**5
NUM_ARG = 200

var_onp = onp.random.rand(1, NUM_VAR)
var = np.asarray(var_onp) # allocating the variables to GRAM
# print(var)
# print(var_onp)
##################################jit
def func(arg):
    divider = 0 # denominator
    numerator = 0
    temp = np.matmul(arg, var)
    temp_sin = np.sin(temp)
    temp_cos = np.cos(temp)
    temp = temp_cos + temp_sin

    temp1 = np.power(temp, 2)
    divider = np.sum(temp1)
    divider = np.power(divider, 1/2)

    temp2 = np.sum(temp, 1)
    temp2 = np.power(temp2, 2)
    numerator = np.sum(temp2)

    ress = np.divide(numerator, divider)

    return np.log(ress)


res = jax.jit(jax.grad(func))

# ##################################autograd
def funcc(arg):
    divider = 0 # denominator
    numerator = 0
    temp = anp.matmul(arg, var)
    temp_sin = anp.sin(temp)
    temp_cos = anp.cos(temp)
    temp = temp_cos + temp_sin

    temp1 = anp.power(temp, 2)
    divider = anp.sum(temp1)
    divider = anp.power(divider, 1/2)

    temp2 = anp.sum(temp, 1)
    temp2 = anp.power(temp2, 2)
    numerator = anp.sum(temp2)

    ress = anp.divide(numerator, divider)

    return anp.log(ress)


ress = autograd.grad(funcc)

# arg = jax.random.normal(jax.random.PRNGKey(1), (NUM_ARG, 0))

arg = onp.random.rand(NUM_ARG, 1)
real_args = np.asarray(arg)
print("... ... ...Start calculating Gradient... ... ...")

# jit
np.set_printoptions(precision=16)
# anp.set_printoptions(precision=16)
start = time.time()
print(res(real_args))
print(type(res(real_args)))
end = time.time()
print("First Execution time using jit: ", end-start, " s ... ... ...")

# # autograd
# start = time.time()
# print(ress(arg))
# end = time.time()
# print("First Execution time using autograd: ", end-start, " s ... ... ...")

for step in range(10):
    arg = onp.random.rand(NUM_ARG, 1)
    real_args = np.asarray(arg)
    # print("... ... ...Start calculating Gradient... ... ...")
    start = time.time()
    res(real_args).block_until_ready()
    end = time.time()
    print(str(step + 1) + "th Execution time on jit: ", end-start, " s ... ... ...")
    # start = time.time()
    # print(ress(arg))
    # end = time.time()
    # print(str(step) + "th Execution time on autograd: ", end-start, " s ... ... ...")