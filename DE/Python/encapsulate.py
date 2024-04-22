class Person:
    def __init__(self, name, age, gender):
        self.__name = name # Private
        self.__age = age
        self.__gender = gender
    
    @property
    def Name(self):
        return self.__name
    
    @Name.setter
    def Name(self, value):
        self.__name = value
    
    # No need for static decorator.. but best practice
    @staticmethod
    def mymethod():
        print("Hello World")

Person.mymethod()

p1 = Person("Tarun", 39, 'M')
# Error - print(p1.__name)
print(p1.Name)

p1.mymethod()
