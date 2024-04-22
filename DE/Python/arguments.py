# python3 arguments.py result.txt -o test.txt -d
def myfunction(*args, **kwargs):
    print(args[0])
    print(args[1])
    print(kwargs['KEYONE'])

myfunction("Hello", True, 10, KEYONE=False)

import sys

# Positional Arguments
print(sys.argv[0]) # Filename
print(sys.argv[1])

# -p 80 // Optional Arguments.

import getopt
opts, orgs = getopt.getopt(sys.argv[1:], "f:m:d:")
# opts will have a list of key and value.
# args will contain arguments with -X

for opt, arg in opts:
    if opt == '-f':
        filename = arg
    if opt == '-m':
        message = arg