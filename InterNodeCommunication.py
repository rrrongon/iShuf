from mpi4py import MPI
import random
import numpy as np
import math 

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
        self.cp_rng.shuffle(self.comm_targets)

        print("initialization done from process #{0}".format(self.rank))

    def _set_send(self, send_request):
        self.send_request = send_request
    
    def _set_recv(self, recv_request):
        self.recv_request = recv_request

    def _communicate(self, batch_index=0):
        send_requests = []
        recv_requests = []
        buf = np.zeros(1<<22,dtype=np.uint8)

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

        for idx in range(0,shuffle_count):
            self.cp_rng.shuffle(self.comm_targets)
            target_rank = self.comm_targets[self.rank] #We can fix target rank == self rank issue here.
            target_ranks.append(target_rank)

        # 2. create random sequence of data (Here we only want the data index sequence. Not shuffle the actual data)
        sample_indexes = list()
        for idx in range(0,shuffle_count):
            sample_index = self.permutation[idx]
            sample_indexes.append(sample_index)

        # 3. Create batch of data
        sample_batch = list()
        for idx in range(0,shuffle_count):
            sample = self.dataset[sample_indexes[idx]]
            sample_batch.append(sample)
            # which data to send?
            # methods 1: shuffle randomly
            # method 2: external sequence

            # method 1:
        # 4. send data
        for idx in range(0,shuffle_count):
            target_rank = target_ranks[idx]
            if target_rank != self.rank:
                #sample = self.dataset[self.permutation[idx]]
                sample = sample_batch[idx]
                #assert sample == sample2 , "sample did not match"

                send_data = {'idx': idx, 'sample': sample}
                req = self.comm.isend(send_data, dest=target_rank, tag=idx)
                print(f"Process #{self.rank}, sending message to {target_rank} with tag {idx}: {send_data}")
                send_requests.append(req)
                self.clean_list.append(self.permutation[idx])

                req = self.comm.irecv(buf, source=MPI.ANY_SOURCE, tag=idx)
                data = req.wait()
                print(f"Process #{self.rank}, Received message : idx {data['idx']} and sample {data['sample']}")
                
                recv_requests.append(req)
            

            send_requests.clear()
            recv_requests.clear()

        return send_requests, recv_requests