from mpi4py import MPI
import random
import numpy as np
import math 
import torch
import sys
import pickle

class ImageNetNodeCommunication:
    def __init__(self, dataset, local_batch_size = 0, fraction = 0, seed = 0):
        self.dataset = dataset
        self.local_batch_size = local_batch_size
        self.fraction = fraction
        self.seed = seed
        self.comm = MPI.COMM_WORLD.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        random.seed(self.seed)
        self.permutation = list(range(len(dataset)))
        random.shuffle(self.permutation)
        self.clean_list = []
        self.cp_rng = np.random.RandomState(seed)
        self.comm_targets = list(range(self.size))
        #self.cp_rng.shuffle(self.comm_targets)
        self.send_requests = list()
        self.recv_requests = list()
        self.sources = list()
        self.recvd_samples = list()

    def _set_send(self, send_request):
        self.send_request = send_request
    
    def _set_recv(self, recv_request):
        self.recv_request = recv_request

    def _communicate(self, batch_index=0):
        if self.fraction ==0:
            return None, None

        shuffle_count = math.floor(len(self.dataset) * self.fraction)

        min_shuffle_count = self.comm.allreduce(shuffle_count, op=MPI.MIN)

        #1. Create the random sequence of target
        target_ranks = list()

        #if self.rank == 0:
        #    print("batch size:{0}. fraction:{1} -> shuffle count {2}".format(self.local_batch_size, self.fraction, min_shuffle_count))
        #    sys.stdout.flush()

        for idx in range(0, min_shuffle_count):
            self.cp_rng.shuffle(self.comm_targets)
            target_rank = self.comm_targets[self.rank]
            target_ranks.append(target_rank)
        
        self.comm.Barrier()

        #print("Rank#{0}, Target ranks len:{1}".format(self.rank, len(target_ranks)))
        #sys.stdout.flush()
        
        sample_indexes = list()
        for idx in range(0, min_shuffle_count):
            sample_index = self.permutation[idx]
            sample_indexes.append(sample_index)
        self.comm.Barrier()

        # 3. Create batch of data
        sample_batch = list()
        for idx in range(len(target_ranks)):
            data_pack = dict()
            data_tup = self.dataset[sample_indexes[idx]]
            data_pack['sample'] = data_tup[0]
            data_pack['label'] = data_tup[1]
            data_pack['path'] = data_tup[2]

            sample_batch.append(data_pack)
            
        self.comm.Barrier()

        for idx in range(len(target_ranks)):
            
            target_rank = target_ranks[idx]
            if self.rank != target_rank:
                #print("Rank#{0}, sending Rank:{1}".format(self.rank, target_rank))
                #sys.stdout.flush()
                data_pack = sample_batch[idx]
                
                send_data = {'idx': idx, 'sample': data_pack['sample'], 'label': data_pack['label'], 'path': data_pack['path']}
                req = self.comm.isend(send_data, dest= target_rank, tag=idx)
                
                self.send_requests.append(req)                
                self.clean_list.append(self.permutation[idx])

                status = MPI.Status()
                buff = np.zeros(1 << 20, dtype=np.uint8)
                recv_req = self.comm.irecv(buff, source=MPI.ANY_SOURCE, tag=idx)

                #source_rank = status.Get_source()
                #received_data = pickle.loads(buff)
                self.recvd_samples.append(recv_req)
            
        self.comm.Barrier()
        
        if self.send_requests is not None and len(self.send_requests)>0:
            for req in self.send_requests:
                req.wait()
        
        self.comm.Barrier()

    def sync_recv(self):
        if self.fraction == 0:
            return
        count = 0
        if self.recvd_samples is not None and len(self.recvd_samples) > 0:
            try:
                #if self.rank ==0:
                #    print("received sample length: {0}".format(len(self.recvd_samples)))
                #    sys.stdout.flush()
                recv_datasamples = list()
                for recv_req in self.recvd_samples:
                    recv_datasamples.append(recv_req.wait())

                self.dataset.add_new_samples(self.rank, recv_datasamples)
                self.recvd_samples.clear()
            except Exception as e:
                print("Exception in rank {0} and error# {1}".format(self.rank, str(e)))
                sys.stdout.flush()
                if self.rank == 0:
                    self.comm.Abort()

    def scheduling(self, epoch):
        if self.fraction == 0:
            return

        random.seed(self.seed + epoch)
        random.shuffle(self.permutation)
    
    def sync_send(self):
        if self.send_requests is not None and len(self.send_requests) > 0:
            count=0
            for idx, req in enumerate(self.send_requests):
                req.wait()
            self.send_requests.clear()


    def clean_sent_samples(self):
        self.dataset.remove_old_samples(self.rank, self.clean_list)
        self.clean_list.clear()

    def get_dataset(self):
        return self.dataset
