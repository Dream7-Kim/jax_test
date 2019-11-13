import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as onp

def readres(fname):
    res = []
    f = open(fname, 'r')
    temp = f.readline()
    strres = temp.split(", ")
    for temp in strres:
        if(temp != ""):
            res.append(float(temp))
    return onp.array(res)

jaxres = readres('jaxres.txt')
laxres = readres('laxres.txt')
autores = readres('autores.txt')

jaxaccu = onp.subtract(jaxres, autores)
jaxaccu = onp.fabs(jaxaccu)
jaxaccu = onp.divide(jaxaccu, autores)
jaxaccu = list(jaxaccu)
print("Accuracy of jax method")
for i in jaxaccu:
    if(i > 1):
        print("****************************")
    print(i)

laxaccu = onp.subtract(laxres, autores)
laxaccu = onp.fabs(laxaccu)
laxaccu = onp.divide(laxaccu, autores)
laxaccu = list(laxaccu)

print("\n\n\n")
print("Accuracy of lax method")
for i in laxaccu:
    if(i > 1):
        print("****************************")
    print(i)

x = onp.arange(len(jaxaccu))

plt.figure(figsize=(21, 5), dpi = 360)
plt.subplot(1, 2, 1)
plt.plot(x, jaxaccu, 'r*', label="jax accu", markersize = 3)
# plt.plot(x, laxaccu, 'bo', label="lax accu", markersize = 0.1)
plt.title("Absolute Error Analysis")
plt.legend(loc='upper right')
plt.xlabel("sub-points")
plt.ylabel("Absolute Error")
plt.subplot(1, 2, 2)
# plt.plot(x, jaxaccu, 'r*', label="jax accu", markersize = 1.0)
plt.plot(x, laxaccu, 'bo', label="lax accu", markersize = 3)
plt.title("Absolute Error Analysis")
plt.legend(loc='upper right')
plt.xlabel("sub-points")
plt.ylabel("Absolute Error")

plt.savefig('res.png', dpi=360)
