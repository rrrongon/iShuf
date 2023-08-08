def read_classes( wnids_file, words_file):
    with open(wnids_file, 'r') as file:
        wnids = [line.strip() for line in file]

    with open(words_file, 'r') as file:
        class_names = [line.strip() for line in file]

    return class_names, wnids


def dump_label(class_labels):
    image_paths = []
    labels = []

    file_path = "./imagenet_dataset/imagenet-mini/class-label.txt"
    with open(file_path, 'w') as file:
        for class_label, wnid in enumerate(class_labels):
            file.write(f"{wnid} {class_label}\n")

def main():
    wnids_file_path = './imagenet_dataset/imagenet-mini/subfolders.txt'   # Replace with the actual path to wnids.txt
    words_file_path = './imagenet_dataset/imagenet-mini/words.txt'   # Replace with the actual path to words.txt

    class_names, wnids = read_classes(wnids_file_path, words_file_path)
    classes, class_labels = read_classes(wnids_file_path, words_file_path)
    print("Class labels {0}".format(class_labels))
    dump_label(wnids)

if __name__ == "__main__":
    main()




