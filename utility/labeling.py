import os, json, csv 

class SampleLabeling:
    def __init__(self, dir, root_dir):
        classes = os.listdir(dir)
        classes.sort()
        self._classes = classes
        self._class_to_label(root_dir, dir)
        self.__create_img_path(root_dir, dir)

    def _class_to_label(self, root_dir, dir):
        _file_names = list(set(dir.split("/"))-set(root_dir.split("/")))
        _file_name = _file_names[0]
        _file_full_path = root_dir + _file_name + "_class_idx.txt"
        _file_full_path_prime = root_dir +_file_name +"_idx_class.txt"

        _isFile = os.path.isfile(_file_full_path)
        if _isFile:
            os.remove(_file_full_path)
        _isFile = os.path.isfile(_file_full_path_prime)
        if _isFile:
            os.remove(_file_full_path_prime)

        f = open(_file_full_path, "w")
        f_prime = open(_file_full_path_prime, "w")

        for idx,_class in enumerate(self._classes):
            f.write( _class+"\t"+str(idx) +"\n")
            f_prime.write(str(idx)+"\t"+_class+"\n")
            
        f.close()
        f_prime.close()

    def __create_img_path(self, root_dir, dir):
        _file_names = list(set(dir.split("/"))-set(root_dir.split("/")))
        _file_name = _file_names[0]
        _file = root_dir + _file_name  + "_filepath.csv"
        _isFile = os.path.isfile(_file)
        if _isFile:
            os.remove(_file)

        fields = ['path', 'label']
        with open(_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for idx,_class in enumerate(self._classes):
                _class_dir = dir + _class + "/"
                if (os.path.isdir(_class_dir)): #also need to check if it is in the label file or not
                    _files = os.listdir(_class_dir)
                    label = idx
                    for _file in _files:
                        _img_path = os.path.join(_class,_file)
                        csvwriter.writerow([_img_path,label])

if __name__ == '__main__':
    f = open('config.json')
    configs =json.load(f)
    root_dir = configs["ROOT_DATADIR"]["train_dir"]
    train_dir = root_dir + "train/"
    SL = SampleLabeling(train_dir, root_dir)
