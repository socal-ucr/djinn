#!/usr/bin/env python2

import numpy as np
import sys
import random

outfile = open("distribution.txt", 'w')
rate = float(sys.argv[1])
total = int(rate) * 120
for i in range(0,200):
    outfile.write(format(random.expovariate(rate),'.6f'))
    outfile.write("\n")

outfile.close()
