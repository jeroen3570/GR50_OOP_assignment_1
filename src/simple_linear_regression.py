import numpy as np

class SimpleLinearRegressor:
    def __init__(self, default_intercept:float, default_slope:float):
        self._intercept = default_intercept
        self._slope = default_slope
    
    def train(self, observations:np.ndarray, target:np.ndarray) -> None:
        '''
        observations is a 1d numpy array of length n
        target is a 1d numpy array of length n
        '''
        obs_mean = observations.mean()
        target_mean = target.mean()
        obs_residuals = observations - obs_mean
        target_residuals = target - target_mean
        optimal_weight = (obs_residuals * target_residuals).sum() / (obs_residuals**2).sum()
        self._slope = optimal_weight
        optimal_bias = target_mean - optimal_weight * obs_mean
        self._intercept = optimal_bias

    def predict(self, data:np.ndarray) -> np.ndarray:
        return self._intercept + self._slope * data

if __name__ == "__main__":
    model = SimpleLinearRegressor(0,0)
    model._intercept = 1.0
    print(f"Model intercept is {model._intercept}")


class Animal:
    def __init__(self, name):
        self.name = name
        self.type = None
        self.crySound = None

    def cry(self):
        print("{} the {} is saying {}".format(self.name, self.type, self.crySound))
    
    def is_eating(self):
        print("{} the {} is eating".format(self.name, self.type))

    def is_sleeping(self):
        print("{} the {} is sleeping".format(self.name, self.type))

    def makes_noise(self):
        print("{} the {} says: {}".format(self.name, self.type, self.crySound))

class Mammal(Animal):
    def __init__(self, name):
        super().__init__(name)
        self.type = type(self).__name__

    def give_offspring(self):
        print("{} the {} is laying an egg.".format(self.name, self.type))

class Oviparous(Animal):
    def __init__(self, name):
        super().__init__(name)
        self.type = type(self).__name__

    def give_offspring(self):
        print("{} the {} is giving birth.".format(self.name, self.type))

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name)
        self.type = type(self).__name__
        self.crySound = "meow"

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)
        self.type = type(self).__name__
        self.crySound = "woof"

class Duck(Animal):
    def __init__(self, name, type):
        super().__init__(name)
        self.type = type(self).__name__
        self.crySound = "quack"

class MagicRock:
    def __init__(self, name):
        self.name = name
        self.type = type(self).__name__

    def is_eating(self):
        print("The magic rock is eating")

    def is_sleeping(self):
        print("The magic rock is sleeping")

def let_it_eat(obj):
    obj.is_eating()

def let_it_sleep(obj):
    obj.is_sleeping()


kate = Mammal("Kate")
kate.is_sleeping()

kitty = Cat("Kitty")
kitty.makes_noise()

rock = MagicRock("RockyRock")
let_it_eat(rock)

print("-----------------")

all_instances = [kate, kitty, rock]

for elem in all_instances:
    let_it_eat(elem)
    let_it_sleep(elem)
    if isinstance(elem, Animal):
        print("It is an animal type")
    else:
        print("It is not an animal type")
    print("\n")

print([instance.name for instance in all_instances]) # list comprehension