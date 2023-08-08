import os

def list_subfolders(folder):
    subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
    return subfolders

def write_subfolders_to_file(subfolders, output_file):
    with open(output_file, 'w') as file:
        for subfolder in subfolders:
            file.write(subfolder + '\n')

if __name__ == "__main__":
    input_folder = "./imagenet_dataset/imagenet-mini/train/"
    output_file = "./imagenet_dataset/imagenet-mini/subfolders.txt"

    subfolders = list_subfolders(input_folder)
    write_subfolders_to_file(subfolders, output_file)
    print("Subfolders written to", output_file)

