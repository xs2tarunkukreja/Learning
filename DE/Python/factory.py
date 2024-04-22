from abc import ABCMeta, abstractstaticmethod

class IPerson(metaclass=ABCMeta):
    @abstractstaticmethod
    def person_method():
        """ Interface Method """

class Student(IPerson):
    def __init__(self):
        self.name = "Basic Student"
    
    def person_method(self):
        print("I am a Student")

class Teacher(IPerson):
    def __init__(self):
        self.name = "Basic Teacher"
    
    def person_method(self):
        print("I am a Teacher")

# Error - p1 = IPerson()

s1 = Student()
s1.person_method()
t1 = Teacher()
t1.person_method()

p1 = None
choice = input("Type: ")
if choice == "Student":
    p1 = Student()
else:
    p1 = Teacher()

class PersonFactory:
    def build_person(choice):
        p1 = None
        if choice == "Student":
            p1 = Student()
        else:
            p1 = Teacher()
        return p1