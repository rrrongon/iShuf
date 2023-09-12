#rm imagenet_dataset/imagenet-mini/parition*.zip
#rm imagenet_dataset/imagenet-mini/parition*.zip
mpirun -np 6 python imagenet_dataPartition.py -npp 6 -f 'imagenet_dataset/imagenet21k_resized/train'

#mpirun -np 4 python imagenet_dataPartition.py -npp 4 -f 'imagenet_dataset/imagenet-mini/train'
mpirun -np 2 python imagenet_partition_copy.py
mpirun -np 1 python imagenet_partition_copy.py
