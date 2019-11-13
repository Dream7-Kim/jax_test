import jax
import jax.numpy as np
import numpy as onp
import time
from jax import lax
import read_data
import write_res

var, arg = read_data.read_data()
var = np.asarray(var)
arg = np.asarray(arg)

def func(arg):
    divider = 0. # NOTE: I made these floats
    numerator = 0.

    def body_fun(carry, x):
        divider, numerator = carry
        temp = np.dot(x, var)
        temp1 = np.sin(temp)
        temp2 = np.cos(temp)

        divid = np.add(temp1, temp2)
        divid = np.power(divid, 2)
        divid = np.sum(divid)

        numer = np.add(temp1, temp2)
        numer = np.sum(numer)
        numer = np.power(numer, 2)
        numerator = np.add(numer, numerator)

        divider = np.add(divider, divid)

        new_carry = divider, numerator
        return new_carry, ()

    (divider, numerator), _ = lax.scan(body_fun, (divider, numerator), arg)
    
    divider = np.power(divider, 1/2)

    return np.log(np.divide(numerator, divider))

res = jax.jit(jax.grad(func))

write_res.writeres(res(arg), 'laxres.txt')