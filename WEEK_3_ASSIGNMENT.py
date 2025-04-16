price = float(input("Enter your price: "))
discount_perct = float(input("Enter your discount percentage: "))

def discount(price, discount_perct):
    return price-(price*discount_perct/100)

discounted_price = discount(price, discount_perct)
print(f"The new price is:{discounted_price:.2f}")