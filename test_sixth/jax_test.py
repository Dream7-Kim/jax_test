import jax
import jax.numpy as np
import numpy as onp
import time
import read_data
import write_res

var, arg = read_data.read_data()
var = np.asarray(var)
arg = np.asarray(arg)
# print(arg.shape)

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

write_res.writeres(res(arg), 'jaxres.txt')

