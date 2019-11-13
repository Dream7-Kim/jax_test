import jax
import jax.numpy as np
import numpy as onp
import time
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

AMOUNT = 10 ** 7
var = onp.zeros(AMOUNT)
for i in range(AMOUNT):
    var[i] = 0.01 * i
# print(var)

def fun(x):
    temp = np.sin(np.dot(x, var))
    return np.sum(temp)

def manual_derivative(x):
    temp = np.cos(np.dot(x, var))
    temp = np.multiply(var, temp)
    return np.sum(temp)

grad = jax.jit(jax.grad(fun))

jaxres = []
manres = []
varx = []
for i in range(10000):
    x = 0.01 * i + 1
    if(i % 100 == 0):
        print("Step " + str(i) + ": ... ... ...")
    varx.append(x)
    jaxres.append(grad(x))
    manres.append(manual_derivative(x))


plt.figure(figsize=(10, 10), dpi = 320)
plt.title("Accuracy Comparation")
plt.xlabel("Independant vairable")
plt.ylabel("Derivative of function cos(x*a)")
plt.plot(varx, jaxres, 'r', label="Result with jax", linewidth = 0.2)
# plt.plot(varx, manres, 'b', label="Result with manual cal")
plt.legend()
plt.savefig('accuracy_comparation_jax.png', dpi=360)
plt.figure(figsize=(10, 10), dpi = 320)
plt.title("Accuracy Comparation")
plt.xlabel("Independant vairable")
plt.ylabel("Derivative of function cos(x*a)")
# plt.plot(varx, jaxres, 'r', label="Result with jax")
plt.plot(varx, manres, 'b', label="Result with manual cal", linewidth = 0.2)
plt.legend()
plt.savefig('accuracy_comparation_man.png', dpi=360)


