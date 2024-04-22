# Magic Method and Dunder
Function starting and ending with __
No need to call them explicitly.

def __init__(self) // Constructor
def __del__(self)  // Destructor

def __add__(self, other) // + Operator Overload

def __str__(self) or __repr__(self) // print(obj)

def __len__(self) // len(obj)

def __call__(self) // obj()

# Decorator
Add additional functionality to a function.

Decorator function return a function that will call the function passed with some other functionality.

# Generator
Generate a sqequence like range function.
Lazy Execution.
It generate part of sequence whenever needed.

yield = Whenever we call yield, it generate next number.

We can create infinite sequence

Don't waste memory.

# Argument Parsing
Any program that run through command line must parse arguments.

# Encapsulation
Data Hiding
Python - Function overloading is not allowed.

# Type Hinting
Python is dynamic typed language.
mypy is something that enforce.

# Factory Design Pattern
code is not working

First solution -
Write a code if and else.

# Proxy Design Pattern
layer of similar object to hide/protect actual object.

# Singleton Design Pattern
only single object of a class.

# Composite Design Pattern
Tree Structure.
Parent type is same as child.
parent can keep list of child of same type as parent.