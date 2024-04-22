from abc import ABCMeta, abstractmethod, abstractstaticmethod

class IDepartment(metaclass=ABCMeta):
    @abstractmethod    
    def __init__(self) -> None:
        """ Implement in child class """
    
    @abstractstaticmethod
    def print_department():
        """ implement in child class """

class Accounting(IDepartment):
    def __init__(self, emplyees) -> None:
        self.employees = emplyees
    
    def print_department(self):
        print(f"Account Department: {self.employees}")

class Development(IDepartment):
    def __init__(self, emplyees) -> None:
        self.employees = emplyees
    
    def print_department(self):
        print(f"Development Department: {self.employees}")

class ParentDepartment(IDepartment):
    def __init__(self, employees) -> None:
        self.employees = employees
        self.base_employees = employees
        self.sub_dept = []
    
    def add(self, dept):
        self.sub_dept.append(dept)
        self.employees += dept.employees
    
    def print_department(self):
        print("Parent Department")
        print(f"Parent Department Base Employees {self.base_employees}")
        for dept in self.sub_dept:
            dept.print_department()
        print(f"total employees {self.employees}")
