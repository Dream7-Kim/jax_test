import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np

colors = ['aliceblue','antiquewhite','aqua','blueviolet','gold','dimgray','red','black','green','blue']
f = open('gpu_time.txt', 'r')
g = open('gpu_time_jit.txt', 'r')
gputime = []
cputime = []
for i in range(10):
    gputime.append([])
    string = f.readline()
    a = string.split(' ')
    # print(a)
    for ele in a:
        if(ele != "\n"):
            # print(ele)
            gputime[i].append(float(ele))
    
    cputime.append([])
    string = g.readline()
    a = string.split(' ')
    # print(a)
    for ele in a:
        if(ele != "\n"):
            # print(ele)
            cputime[i].append(float(ele))

f.close()
g.close()

# print(time)
data_num =[]
for i in range(100):
    data_num.append(int(10**7/100*(i+1)/8))

plt.figure(figsize=(40, 10), dpi = 360)

for i in range(10):
    if(i>0):
        plt.subplot(3, 3, i)
        plt.title("GPU vs GPU_jit Execution Time(" + str(i) + "th execution)")
        plt.plot(np.log10(data_num), gputime[i], color = colors[i], label = str(i) + "th calc on GPU")
        plt.plot(np.log10(data_num), cputime[i], color = colors[i], linestyle='dashed',label = str(i) + "th calc on GPU with jit")
        plt.legend(loc='upper right')
        plt.xlabel("Data Amount(log10)")
        plt.ylabel("Execution time(s)")

plt.savefig('jit_time.png', dpi=360)

    

