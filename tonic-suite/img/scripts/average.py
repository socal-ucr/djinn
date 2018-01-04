#!/usr/bin/python
"""Collect command-line options in a dictionary"""
import sys

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

    sum = 0.0
    for i in content:
        sum += float(i)

    sum /= len(content)

    print(sum)
