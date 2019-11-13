import numpy as np

NUM_VAR = 10**2
NUM_ARG = 200

rng = np.random.RandomState(0)

var = 10 * rng.rand(1, NUM_VAR)

arg = rng.rand(NUM_ARG)

varf = open('variable.txt', 'w')
for vari in var[0]:
    varf.write(str(vari) + " ")
varf.close()

argf = open('argument.txt', 'w')
for argi in arg:
    argf.write(str(argi) + " ")
argf.close()