import autograd
import autograd.numpy as anp
import time
import read_data
import write_res

var, arg = read_data.read_data()

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


res = autograd.grad(funcc)

write_res.writeres(res(arg), 'autores.txt')