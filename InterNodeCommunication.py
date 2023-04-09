from mpi4py import MPI
import random
import numpy as np
import math 
import torch
import sys
import pickle

MPI.MAX_BUFFER_SIZE = 1024 * 1024  # 1 GB

def get_source_rank(myRank, targets):
    source = -1
    for i, rank in enumerate(targets):
        if rank == myRank:
            #print(f"Found {value_to_find} at index {i}")
            source = i
            break
    return source

class NodeCommunication:
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
        #send_requests = []
        #recv_requests = []
        #buf = np.zeros(1<<22,dtype=np.uint8)
        #buf_size = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
        #buf = np.zeros(buf_size, dtype=np.uint8)
        #buf = bytearray(80)
        '''
            1. Create the random sequence of target
            2. create random sequence of data (Here we only want the data index sequence. Not shuffle the actual data)
            3. Create batch of data
            4. send data
        '''

        if self.fraction ==0:
            return None, None
        
        shuffle_count = math.floor(self.local_batch_size * self.fraction)

        #1. Create the random sequence of target
        target_ranks = list()
        source_ranks = list()
        comm_pack = dict()
        comm_packs = list()
        for idx in range(0, shuffle_count):
            self.cp_rng.shuffle(self.comm_targets)
            target_rank = self.comm_targets[self.rank] #We can fix target rank == self rank issue here.
            #source_rank = get_source_rank(self.rank, self.comm_targets)
            #if self.rank != target_rank:
            target_ranks.append(target_rank)
            #source_ranks.append(source_rank)
            #if target_rank!=source_rank:
            #    comm_pack = {'s': source_rank, 't':target_rank}
            #    comm_packs.append(comm_pack)

        self.comm.Barrier()

        print("Rank#{0} target list#{1}".format(self.rank, target_ranks))
        
        #print("Rank#{0} pack list#{1}".format(self.rank, comm_packs))

        # 2. create random sequence of data (Here we only want the data index sequence. Not shuffle the actual data)
        sample_indexes = list()
        for idx in range(0, shuffle_count):
            sample_index = self.permutation[idx]
            sample_indexes.append(sample_index)
        
        self.comm.Barrier()

        # 3. Create batch of data
        sample_batch = list()
        for idx in range(len(target_ranks)):
            data_pack = dict()
            data_tup = self.dataset[sample_indexes[idx]]
            #print(data_tup)
            data_pack['sample'] = data_tup[0]
            data_pack['label'] = data_tup[1]
            data_pack['path'] = data_tup[2]
            sample_batch.append(data_pack)
            # which data to send?
            # methods 1: shuffle randomly
            # method 2: external sequence
        
        print("Rank#{0} batch_size#{1}".format(self.rank, len(sample_batch)))
        self.comm.Barrier()
            # method 1:
        # 4. send data
        for idx in range(len(target_ranks)):
            target_rank = target_ranks[idx]
            if self.rank != target_rank:
                data_pack = sample_batch[idx]
                sample = data_pack['sample']
                label = data_pack['label']
                path = data_pack['path']

                send_data = {'idx': idx, 'sample': sample, 'path': path, 'class_name': label}
                req = self.comm.isend(send_data, dest= target_rank, tag=idx)
                print(f"Process #{self.rank}, sending message to {target_rank} with tag {idx}: ")
                sys.stdout.flush()
                self.send_requests.append(req)
                self.clean_list.append(self.permutation[idx])

                status = MPI.Status()
                buff = np.zeros(1 << 20, dtype=np.uint8)
                self.comm.recv(buff, source=MPI.ANY_SOURCE, tag=idx, status=status)

                # get the rank that sent the data and the data itself
                source_rank = status.Get_source()
                received_data = pickle.loads(buff)

                print(f"Process #{self.rank}, received message from {source_rank} with tag {idx}: ")
                print(received_data)
                sys.stdout.flush()
                #buf = np.zeros(1<<20,dtype=np.uint8)
                #req = self.comm.irecv(buf, source = MPI.ANY_SOURCE, tag=idx)
                #data = req.wait()
                #print("data-----{0}".format(data['path']))
                #buf = np.zeros(buf_size, dtype=np.uint8)
                #req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=idx)
                #status = MPI.Status()
                #req.wait(status=status)
                #source = status.Get_source()
                #data = req.wait()
                #print(f"Process #{self.rank}, Received message : idx {data['idx']} and, label {data['class_name']}, ImagePath {data['path']}")
                #self.sources.append(source)
                #print(f"Process #{self.rank}, receiving message from {source} with tag {idx}: ")
                #self.recv_requests.append(req)
                #self.sync_recv()
                self.recvd_samples.append(received_data)
            
        #return self.send_requests, self.recv_requests

    def sync_recv(self):
        if self.fraction == 0:
            return
        
        count = 0
        if self.recvd_samples is not None and len(self.recvd_samples) > 0:
            for sample in self.recvd_samples:
                try:
                    self.dataset.add_new_samples(self.rank, self.recvd_samples)
                    self.recvd_samples.clear()

                except Exception as e:
                    print("Exception in rank {0} and error# {1}".format(self.rank, str(e)))
                    sys.stdout.flush()
                    if self.rank == 0:
                        self.comm.Abort()

        print("Rank#{0} is done receiving.".format(self.rank))        
        sys.stdout.flush()
    
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

        print("Rank#{0} is done Sending.".format(self.rank))
        sys.stdout.flush()

    def clean_sent_samples(self):
        self.dataset.remove_old_samples(self.rank, self.clean_list)
        self.clean_list.clear()
        print("Rank#{0} is done sleaning.".format(self.rank))
        sys.stdout.flush()
        
