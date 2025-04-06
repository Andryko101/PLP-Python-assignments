file_to_open=input("Enter file to open: ")

try:
    file=open(file_to_open, "r")
    data=file.read()
    print(data)
    file.close()
except FileNotFoundError:
    print("Your File was not found")
