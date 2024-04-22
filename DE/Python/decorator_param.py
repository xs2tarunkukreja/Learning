def mydecorator(function):

    def wrapper(*args, **kwargs):
        ret_val = function(*args, **kwargs)
        print("My New Decorator")
        return ret_val
        

    return wrapper

@mydecorator
def hello(person):
    print(f"Hello {person}")

hello("Tarun")


def logged(function):
    def wrapper(*args, **kwargs):
        value = function(*args, **kwargs)
        with open('logfile.txt', 'a+') as f:
            fname = function.__name__
            f.write(f"{fname} returned value {value}\n")
        return value
    return wrapper

import time
def timed(function):
    def wrapper(*args, **kwargs):
        before = time.time()
        value = function(*args, **kwargs)
        after = time.time()
        fname = function.__name__
        print(f"{fname} took {after-before} second to execute!")
        return value

# What if we want to return something from hello
    # Wrapper return.
# What if we want to call hello first and then call other
    # Save return value in some variable.
    # Do Job and then return variable
# If hello is also printing with returning.
    # No Impact.. Flow will be like wrapper.