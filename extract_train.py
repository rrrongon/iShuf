import os
import tarfile

def extract_and_remove_tar_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a tar file
        if file_name.endswith('.tar'):
            # Create a folder with the same name as the tar file (without the .tar extension)
            extracted_folder = os.path.join(folder_path, os.path.splitext(file_name)[0])
            os.makedirs(extracted_folder, exist_ok=True)

            # Extract the contents of the tar file to the new folder
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(extracted_folder)
            print(f"Extracted contents from {file_name}")

            # Remove the tar file
            os.remove(file_path)
            print(f"Removed {file_name}")

if __name__ == "__main__":
    folder_path = "/home/rongon/Documents/Research/Project_Shuffling/customdatasampler/imagenet_dataset/imagenet21k_resized/train/"  # Replace with the path to your folder containing tar files
    extract_and_remove_tar_files(folder_path)

