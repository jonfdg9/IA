import math as m

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def module(self):
        return m.sqrt(self.x**2 + self.y**2)
    
    def scalar(self, param=1):
        self.x *= param
        self.y *= param
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    @staticmethod
    def sum(v1,v2):
        return Vector2D(v1.x + v2.x, v1.y + v2.y)

    @staticmethod
    def restar(v1,v2):
        return Vector2D(v1.x - v2.x, v1.y - v2.y)
        

class Vector3D(Vector2D):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z=z
    
    def module(self):
        return m.sqrt(self.x**2 + self.y**2,self.z**2)

    def scalar(self,param=1):
        super().scalar(param)
        self.z*=param
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    @staticmethod
    def sum(v1,v2):
        return Vector3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

    @staticmethod
    def restar(v1,v2):
        return Vector3D(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)
    
    @staticmethod
    def product(v1,v2):
        return Vector3D(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
    
    @staticmethod
    def distance(v1, v2):
        return m.sqrt((v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2)