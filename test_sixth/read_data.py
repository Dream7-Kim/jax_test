import numpy as onp
import jax.numpy as np

def read_data():
    varf = open('variable.txt', 'r')
    temp = varf.readline()
    varf.close()
    temp = temp.split(" ")
    var = onp.zeros((1, len(temp)-1), dtype='float64')
    # print(temp)
    itera = 0
    for string in temp:
        if(string != ""):
            var[0, itera] = float(string)
            itera += 1

    # print(var)

    argf = open('argument.txt', 'r')
    temp = argf.readline()
    argf.close()
    temp = temp.split(" ")
    arg = onp.zeros((len(temp)-1, 1), dtype='float64')
    # print(temp)
    itera = 0
    for string in temp:
        if(string != ""):
            arg[itera][0] = float(string)
            itera += 1

    return var, arg
    

# print(read_data())