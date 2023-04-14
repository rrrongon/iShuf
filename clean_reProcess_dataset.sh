#!/bin/bash

cd natural_image/Partition_Folder/
rm -rf *
cd ../..

cd DataProcessing
./data_partition.sh
cd ..

mpirun -n 4 python UnitTest/Test_Partition_download.py
mpirun -n 4 python utility/_dataSplit.py
mpirun -n 4 python utility/labeling.py
