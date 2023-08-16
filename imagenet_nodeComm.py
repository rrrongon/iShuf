from mpi4py import MPI
import random
import numpy as np
import math
import torch
import sys
import pickle
import horovod.torch as hvd
from imp_sampling_handler import ImpSam as IS

class ImageNetNodeCommunication:
    def __init__(self, dataset, local_batch_size = 0, fraction = 0, seed = 0, min_train_dataset_len=0, epochs=0):
        self.dataset = dataset
        self.local_batch_size = local_batch_size
        self.fraction = fraction
        self.seed = seed
        self.comm = MPI.COMM_WORLD.Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        random.seed(self.seed)
        self.permutation = list(range(min_train_dataset_len))
        random.shuffle(self.permutation)
        self.clean_list = []
        self.cp_rng = np.random.RandomState(seed)
        self.comm_targets = list(range(self.size))
        #self.cp_rng.shuffle(self.comm_targets)
        self.send_requests = list()
        self.recv_requests = list()
        self.sources = list()
        self.recvd_samples = list()
        self.iSample = IS(epochs)

    def _set_current_epoch(self, epoch):
        self.iSample.set_current_spoch(epoch)

    def _set_current_unsorted_batchLoss(self, unsorted_data):
        self.iSample.set_current_unsorted_batchLoss(unsorted_data)

    def _set_send(self, send_request):
        self.send_request = send_request

    def _set_recv(self, recv_request):
        self.recv_request = recv_request

    def _communicate(self, batch_index=0):
        if self.fraction ==0:
            return None, None

        shuffle_count = math.floor(len(self.dataset) * self.fraction)
        shuffle_count = torch.tensor(shuffle_count)

        min_shuffle_count = hvd.allreduce(shuffle_count, op=hvd.mpi_ops.Min)
        min_shuffle_count = min_shuffle_count.item()

        print("Node comm min finding complete. Min shuffle count#{0}".format(min_shuffle_count))
        sys.stdout.flush()

        #min_shuffle_count = self.comm.allreduce(shuffle_count, op=MPI.MIN)

        #1. Create the random sequence of target
        target_ranks = list()

        #if self.rank == 0:
        #    print("batch size:{0}. fraction:{1} -> shuffle count {2}".format(self.local_batch_size, self.fraction, min_shuffle_count))
        #    sys.stdout.flush()

        for idx in range(0, min_shuffle_count):
            self.cp_rng.shuffle(self.comm_targets)
            target_rank = self.comm_targets[self.rank]
            target_ranks.append(target_rank)

        #self.comm.Barrier()
        hvd.allreduce(torch.tensor(0), name="barrier")

        #print("Rank#{0}, Target ranks len:{1}".format(self.rank, len(target_ranks)))
        #sys.stdout.flush()

        # Get important sampling
        top_percent = math.ceil(self.fraction * 100) + 3
        important_samples = self.iSample.get_top_x_sample(top_percent) #important samples as a dict

        #if self.rank==0:
        #    print("Important Samples: in rank {0}: {1}\n".format(self.rank, important_samples))
        #    sys.stdout.flush()

        sample_indexes = list()
        #for idx in range(0, min_shuffle_count):
        for sample_idx, loss in important_samples.items():
            #sample_idx = self.permutation[idx]
            sample_indexes.append(sample_idx)
            #if self.rank==0:
            #    print("Sending sample index: {0}".format(sample_idx))

        del important_samples

        #self.comm.Barrier()
        hvd.allreduce(torch.tensor(0), name="barrier")

        # 3. Create batch of data
        #sample_batch = list()
        #for idx in range(len(target_ranks)):
        #    data_pack = dict()

        #    data_tup = self.dataset[sample_indexes[idx]]
        #    data_pack['sample'] = data_tup[0]
        #    data_pack['label'] = data_tup[1]
        #    data_pack['path'] = data_tup[2]

        #    sample_batch.append(data_pack)

        #self.comm.Barrier()
        #hvd.allreduce(torch.tensor(0), name="barrier")

        for idx in range(len(target_ranks)):

            target_rank = target_ranks[idx]
            if self.rank != target_rank:
                #print("Rank#{0}, sending Rank:{1}".format(self.rank, target_rank))
                #sys.stdout.flush()
                data_pack = dict()

                data_tup = self.dataset[sample_indexes[idx]]
                data_pack['sample'] = data_tup[0]
                data_pack['label'] = data_tup[1]
                data_pack['path'] = data_tup[2]

                #data_pack = sample_batch[idx]
                #sample_batch[idx] = None

                send_data = {'idx': idx, 'sample': data_pack['sample'], 'label': data_pack['label'], 'path': data_pack['path']}
                req = self.comm.isend(send_data, dest= target_rank, tag=idx)
                del data_pack
                del send_data
                del data_tup

                self.send_requests.append(req)
                self.clean_list.append(sample_indexes[idx])

                status = MPI.Status()
                buff = np.zeros(1 << 20, dtype=np.uint8)
                recv_req = self.comm.irecv(buff, source=MPI.ANY_SOURCE, tag=idx)

                print("Rank#{0} sending..".format(self.rank))
                #source_rank = status.Get_source()
                #received_data = pickle.loads(buff)
                self.recvd_samples.append(recv_req)
                del buff
                del recv_req

                if self.send_requests is not None and len(self.send_requests)>0:
                    for req in self.send_requests:
                        req.wait()

                if self.recvd_samples is not None and len(self.recvd_samples) > 0:
                    try:
                        #if self.rank ==0:
                        #    print("received sample length: {0}".format(len(self.recvd_samples)))
                        #    sys.stdout.flush()
                        recv_datasamples = list()
                        for recv_req in self.recvd_samples:
                            recv_datasamples.append(recv_req.wait())
                            del recv_req

                        self.dataset.add_new_samples(self.rank, recv_datasamples)
                        self.recvd_samples.clear()
                        del self.recvd_samples
                        del recv_datasamples
                        self.recvd_samples = list()
                        print("Rank#{0} receiving..".format(self.rank))
                    except Exception as e:
                        print("Exception in rank {0} and error# {1}".format(self.rank, str(e)))
                        sys.stdout.flush()
                        if self.rank == 0:
                            self.comm.Abort()

        print("Node comm all sending complete")
        sys.stdout.flush()

        #self.comm.Barrier()
        hvd.allreduce(torch.tensor(0), name="barrier")

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
                    del recv_req

                self.dataset.add_new_samples(self.rank, recv_datasamples)
                self.recvd_samples.clear()
                del recv_datasamples

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
            del self.send_requests
            self.send_requests = list()


    def clean_sent_samples(self):
        self.dataset.remove_old_samples(self.rank, self.clean_list)
        self.clean_list.clear()
        del self.clean_list
        self.clean_list = list()

    def get_dataset(self):
        return self.dataset

    def dump_result(self, rank):
        file_name = 'loss_values_'+str(rank)+'.json'
        self.iSample.save_loss_values_excel(file_name)
