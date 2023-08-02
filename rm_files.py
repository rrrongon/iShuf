import os

def remove_files_in_directory(directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Iterate through each file and remove it
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)

        # Check if it is a file (not a directory) and remove it
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed {file_name}")

if __name__ == "__main__":
    directory_path = "imagenet_dataset/imagenet21k_resized/train/"  # Replace with the path to your directory
    remove_files_in_directory(directory_path)

