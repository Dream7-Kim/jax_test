import jax
import jax.numpy as np
import numpy as onp
import time
import matplotlib.pyplot as plt

gpu_time = []
gpu_time1 = []
gpu_time2 = []

for step in range(100):
    start = time.time()
    NUM = int(10**7/100*(step+1))
    args = onp.zeros([NUM, 3])
    for i in range(NUM):
        args[i, ] = [1000/NUM*i - 2, 100/NUM*i - 4, 10/NUM*i - 6]
    args = np.asarray(args)
    end = time.time()
    # print("Times to allocating memory: ", end - start, " s")
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
    grad = jax.jacrev(func)

    # print("\n\n\n")
    print("Step: \t", step, "Inner Step: 1")
    start = time.time()    
    print("Gradient: ", grad(np.array([1.1, 1.5, 1.9])))
    end = time.time()
    print("First execution time in this step: ", end-start, " s   ")
    gpu_time.append(float(end-start))
    # print("\n\n\n")
    print("Step: \t", step, "Inner Step: 2")
    start = time.time()    
    print("Gradient: ", grad(np.array([2.1, 2.5, 2.9])))
    end = time.time()
    gpu_time1.append(float(end-start))
    # print("\n\n\n")
    print("Step: \t", step, "Inner Step: 3")
    start = time.time()    
    print("Gradient: ", grad(np.array([3.1, 3.5, 3.9])))
    end = time.time()
    gpu_time2.append(float(end-start))


print(gpu_time)
print(gpu_time1)
print(gpu_time2)
write_date = open('gpu_time.txt', 'w')
write_date.writelines(gpu_time)
write_date.writelines(gpu_time1)
write_date.writelines(gpu_time2)
write_date.close()

print("------------------------------Finished------------------------------")


