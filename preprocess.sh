#cd DataProcessing
#./data_partition.sh
#cd ..
#mpirun -n 4 python UnitTest/Test_Partition_download.py
#mpirun -n 4 python utility/_dataSplit.py
#mpirun -n 4 python utility/labeling.py

#mpirun -np 2 python imagenet_dataPartition.py -npp 2 -f 'imagenet_dataset/imagenet21k_resized/train'
mpirun -np 1 python imagenet_partition_copy.py
horovodrun -np 2 -H localhost:2 python dataloader_test.py
