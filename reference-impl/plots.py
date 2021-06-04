import math
import numpy as np
import matplotlib.pyplot as plt
import sys

def graph(func, x_range):
   # x = np.arange(*x_range)
   # y = func(x)

   x = np.linspace(sys.float_info.min, 1.0, 100)
   y = np.exp(-1/x)
   plt.figure()
   plt.plot(x, y)
   plt.title('address stake vs bandwidth')
   plt.xlabel('DAG in address / of total DAG in circulation')
   plt.ylabel('% stake of total bandwidth (transactions/sec)')

if __name__ == '__main__':
    graph(lambda x: np.power(-1.0, x), (0.0, 100.0, 0.01))
    t = ""
