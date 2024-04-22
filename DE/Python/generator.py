def mygenerator(n):
    for x in range(n):
        yield x**3

values = mygenerator(10) # Values have a generator which will provide values whenever needed.

print(next(values))
print(next(values))
print(next(values))

for x in values: 
    print(x)

def infinite_sequence():
    result = 1
    while True:
        yield result
        result *= 5