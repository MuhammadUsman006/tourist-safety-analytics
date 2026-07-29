import os

# Show us exactly where Python is currently "standing"
print("Current working directory:")
print(os.getcwd())
print()

# List everything in that folder
print("Files in current working directory:")
for item in os.listdir("."):
    print(" -", item)