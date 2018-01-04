#!/usr/bin/python
"""Collect command-line options in a dictionary"""
import sys
from math import ceil

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ == '__main__':
    from sys import argv
   
    myargs = getopts(argv)
    fname = myargs['-i']


    
    with open(fname) as f:
        content = f.readlines()

    data = []
    for i in content:
        data.append(float(i))

    data = sorted(data)

    num = float(len(data))
    percentile = int(ceil(num * .99))
    print(data[percentile -1])

