#!/usr/bin/python

import numpy as np
import sys

rate = float(sys.argv[1])
sample = np.random.poisson(rate, 1000)

outfile = open("distribution.txt", 'w')
out = []
for element in sample:
    if(element != 0):
        outfile.write(str(1/float(element)))
        outfile.write("\n")
    else:
        outfile.write("0\n")


outfile.close()
