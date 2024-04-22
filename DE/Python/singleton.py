from abc import ABCMeta, abstractstaticmethod
class IApplication(metaclass=ABCMeta):

    @abstractstaticmethod
    def print_data():
        """ Implement in child class """

class Application(IApplication):
    
    __instance = None

    @staticmethod
    def get_instance():
        if Application.__instance == None:
            Application.__instance = Application()
        
        return Application.__instance
    
    def __init__(self):
        if (Application.__instance != None):
            raise Exception("Singleton can't be initiated more than once.")
            
        Application.__instance = self
    
    @staticmethod
    def print_data():
        print("Singleton instance")
        
