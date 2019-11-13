import autograd
import autograd.numpy as np
import numpy as onp
import time
from autograd import jacobian

f = open('gradient_autograd.txt', 'w')

for step in range(100):
    f.write("Step " + str(step) + "\n")
    print("Step:", str(step))
    start = time.time()
    NUM = int(10**7/100*(step+1)/8)
    args = onp.zeros([NUM, 3])
    for i in range(NUM):
        args[i, ] = [1000/NUM*i - 2, 100/NUM*i - 4, 10/NUM*i - 6]
    # args = np.asarray(args)
    end = time.time()
    print("\t\tTimes to allocating memory: ", end - start, " s")
    # print(args)
    # print(np.multiply(args[:,0], args[:,1]))

    def func(var):
        # (var[0]-args[:,0])**3
        temp = np.sin(np.power(np.subtract(var[0], args[:, 0]), 3))
        ele1 = np.sum(temp)
        # sin(var[1]**3 - var[2]**2) * args[:,1]
        temp = np.sin(np.subtract(np.power(var[1], 3), np.power(var[2], 2)))
        ele2 = np.sum(np.dot(temp, args[:, 1]))
        # log(args[:,1]*args[:,2]*var[0]*var[1]*var[2])
        temp = np.multiply(args[:, 1], args[:, 2])
        ele3 = np.sum(np.log(1 + np.abs(np.dot(var[0] * var[1] * var[2], temp))))
        # print("\n\n", ele1, "\n", ele2, "\n", ele3)
        # print(res, type(res))
        return ele1 + ele2 + ele3


    # print("-------------Reverse mode---------------")
    # jax.jit(func)
    grad = jacobian(func)

    for j in range(10):
        f.write("\tSubstep " + str(j+1) + "\n")
        var = np.array([0.001*j, 0.001*2*j, 0.001*3*j])
        print("\tSubstep:", str(j))
        start = time.time()
        gradient = grad(var)
        end = time.time()
        f.write("\t\t" + str(gradient) + "\n")
        print("\t\tGradient:", gradient)
        print("\t\tExecution time:", float(end-start))
        



# for ele in exetime:
#     for x in ele:
#         f.write(str(x)+" ")
#     f.write("\n")

f.close()


print("------------------------------Finished------------------------------")