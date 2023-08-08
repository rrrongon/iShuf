#rm imagenet_dataset/imagenet-mini/parition*.zip
#rm imagenet_dataset/imagenet-mini/parition*.zip
#mpirun -np 7 python imagenet_dataPartition.py -npp 7 -f 'imagenet_dataset/imagenet21k_resized/train'

mpirun -np 4 python imagenet_dataPartition.py -npp 4 -f 'imagenet_dataset/imagenet-mini/train'
mpirun -np 4 python imagenet_partition_copy.py
