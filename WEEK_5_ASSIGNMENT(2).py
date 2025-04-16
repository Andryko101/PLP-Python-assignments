class creature():
    def __init__(self, name, legs, movement):
        self.name=name
        self.legs=legs
        self.movement=movement

    def moving(self):
        print(f"{self.name} has {self.legs} legs and moves by {self.movement}")

class insect(creature):
    def __init__(self, name, legs):
        super().__init__(name, int(legs), "crawling")

class fish(creature):
    def __init__(self, name):
        super().__init__(name, 0, "swimming")
   
class mammal(creature):
   def __init__(self, name, legs,):
        super().__init__(name, int(legs), "walking")

class bird(creature):
    def __init__(self, name):
        super().__init__(name, 2, "flying")

organism=(input("What type of organism is this?(insect, mammal, fish or bird): "))
if organism=="insect":
    name=(input("Enter the name of the Insect: "))
    legs=(input (int("Enter the number of legs the Insect has: ")))
    my_organism=insect(name, legs)
    my_organism.moving()
elif organism=="fish":
    name=(input("Enter the name of the fish: "))
    my_organism=fish(name)
    my_organism.moving()
elif organism=="mammal":
    name=(input("Enter the name of the mammal: "))
    legs=(input (int("Enter the number of legs that the mammal has: ")))
    my_organism=mammal(name, legs)
    my_organism.moving()
elif organism=="bird":
    name=(input("Enter the name of the bird: "))
    my_organism=bird(name)
    my_organism.moving()
else:
    print("Kindly choose from the list of animals available")
    