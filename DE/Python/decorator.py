# It is an idea. But not a way to achieve in Python.
def mydecorator(function):

    def wrapper():
        print("I am decorating your function")
        function()
    
    return wrapper
    
def hello_world():
    print("Hello World")

mydecorator(hello_world)()

# Actual Implementation in Python

@mydecorator
def hello_world_v2():
    print("hello world")

hello_world_v2() # Here actual decorator function is called.
