#rm imagenet_dataset/imagenet-mini/parition*.zip
#rm imagenet_dataset/imagenet-mini/parition*.zip
mpirun -np 8 python imagenet_dataPartition.py -npp 20 -f 'imagenet_dataset/imagenet21k_resized/train'

#mpirun -np 5 python imagenet_dataPartition.py -npp 10 -f 'imagenet_dataset/imagenet-mini/train'
mpirun -np 8 python imagenet_partition_copy.py
