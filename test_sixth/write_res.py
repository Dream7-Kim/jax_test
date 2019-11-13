import jax.numpy as np

def writeres(res, fname):
    f = open(fname, 'w+')
    for i in res:
        f.write(str(i[0]) + ", ")
    f.close()
