class superhuman:
    def __init__(self, name, planet, power, weakness):
        self.name=name
        self.planet=planet
        self.power=power
        self.__weakness=weakness

    def characters(self):
        print(f"My name is {self.name}. I am the protector of {self.planet}. My main power is{self.power}")
    
    def secret_weakness(self):
        print(f"My secret weakness is {self.__weakness}")

class superhero(superhuman):
    pass

class supervillain(superhuman):
    pass
    def characters(self):
        print(f"My name is {self.name}. I am {self.planet}'s worst nightmare. I will destroy {self.power}")
    
identity=(input("Are you a villan or a hero: "))
if identity=="hero":
    name=(input("Enter the hero name: "))
    planet=(input("Enter the hero home planet: "))
    power=(input("Enter the hero power: "))
    weakness=(input("Enter your secret weakness: "))
    hero=superhero(name, planet, power, weakness)
    hero.characters()
elif identity=="villain":
    name=(input("Enter the villain name: "))
    planet=(input("Enter the villain planet for carnage: "))
    power=(input("Enter the villain target: "))
    weakness=(input("Enter your secret weakness: "))
    hero=supervillain(name,planet,power, weakness)
    hero.characters()
else:
    print("Choose either the value hero or villain")


view_weakness=(input("Would you like to view your secret weakness?(Yes/No)"))
if view_weakness=="Yes":
    hero.secret_weakness()
elif view_weakness=="No":
    print("Okay")
else:
    print("Select either Yes or No!")