import os

# Define the path to your directory
path = "/hpc2hdd/home/lwang851/Project/SAFMN/datasets/DF2K/DF2K_train_LR_bicubic/X3"

# Iterate over all files in the directory
for filename in os.listdir(path):
    if filename.endswith(".png") and filename.endswith("x3.png"):
        # Construct the full file path
        full_path = os.path.join(path, filename)
        # Remove the last "x4" before the file extension
        new_filename = filename.replace("x3.png", ".png")
        new_full_path = os.path.join(path, new_filename)
        # Rename the file
        os.rename(full_path, new_full_path)

print("Files have been renamed.")