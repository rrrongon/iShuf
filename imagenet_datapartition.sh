#mpirun -np 4 python imagenet_dataPartition.py -npp 4 -f 'imagenet_dataset/imagenet21k_resized/train'
mpirun -np 4 python imagenet_dataPartition.py -npp 4 -f 'imagenet_dataset/imagenet-mini/train'
