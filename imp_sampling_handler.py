import math
from itertools import islice

import json

def get_top_epochs(index, temp_results, start_epoch=0):
    result = list()
    for epoch, top_result in enumerate(temp_results):
        if index in top_result:
            loss = top_result[index]
            result.append((start_epoch+epoch,loss))

    return result

class ImpSam:
    def __init__(self, epoch):
        self.epoch = epoch
        self.sorted_hist_batchLoss = []
        self.current_epoch = 0
        self.current_unsorted_batchLoss = dict()
        self.top_percent = 10
        self.dataset_size = 0

    def set_current_spoch(self, epoch_no):
        self.current_epoch = epoch_no

    def set_current_unsorted_batchLoss(self, us_batchLoss):
        self.current_unsorted_batchLoss = us_batchLoss
        self.sync()

    def sync(self):
        current_sorted_batchLoss = dict(sorted(self.current_unsorted_batchLoss.items(), key=lambda x: x[1], reverse=True))
        self.sorted_hist_batchLoss.append(current_sorted_batchLoss)
        self.dataset_size = len(current_sorted_batchLoss)

    def get_top_x_sample(self, top_percent=None, which_epoch=None):
        if which_epoch==None:
            index = self.current_epoch
        else:
            index = which_epoch

        if top_percent is None:
            top_n = math.ceil((self.top_percent/100) * self.dataset_size)
        else:
            top_n = math.ceil((top_percent/100) * self.dataset_size)

        epoch_sorted_batchLoss = self.sorted_hist_batchLoss[index]
        top_n_elements = dict(islice(epoch_sorted_batchLoss.items(), top_n))

        return top_n_elements


    def get_nEpochs_top_x_samples(self, start_epoch=0, end_epoch=0, top_percent=None):
        # Extract the first two dictionaries and sort them by values
        sorted_dicts = sorted(self.sorted_hist_batchLoss[start_epoch:end_epoch], key=lambda d: sum(d.values()))

        # Calculate the average for each key across the sorted dictionaries
        averages = {key: sum(d[key] for d in sorted_dicts) / len(sorted_dicts) for key in sorted_dicts[0]}

        # Create a new dictionary with the same keys in sorted order and the calculated average values
        result = {key: averages[key] for key in sorted(averages, key=averages.get, reverse = True)}

        if top_percent is None:
            top_n = math.ceil((self.top_percent/100) * self.dataset_size)
        else:
            top_n = math.ceil((top_percent/100) * self.dataset_size)

        top_n_elements = dict(islice(result.items(), top_n))

        return top_n_elements


    def get_global_topX_samples(self, top_percent=None):
        start_epoch = 0
        end_epoch = self.current_epoch

        if top_percent is None:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch)
        else:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch, top_percent=top_percent)

        return result

    def get_range_topX_samples(self, start_epoch, end_epoch, top_percent = None):
        if top_percent is None:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch)
        else:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch, top_percent=top_percent)

        return result

    def get_list_topX_samples(self, epoch_list, top_percent = None):
        sorted_dicts = [self.sorted_hist_batchLoss[i] for i in epoch_list]

        # Calculate the average for each key across the sorted dictionaries
        averages = {key: sum(d[key] for d in sorted_dicts) / len(sorted_dicts) for key in sorted_dicts[0]}

        # Create a new dictionary with the same keys in sorted order and the calculated average values
        result = {key: averages[key] for key in sorted(averages, key=averages.get, reverse = True)}

        if top_percent is None:
            top_n = math.ceil((self.top_percent/100) * self.dataset_size)
        else:
            top_n = math.ceil((top_percent/100) * self.dataset_size)

        top_n_elements = dict(islice(result.items(), top_n))
        return top_n_elements

    def get_lastN_topX_samples(self, n, top_percent = None):
        start_epoch = self.current_epoch - n
        end_epoch = self.current_epoch
        if top_percent is None:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch)
        else:
            result = self.get_nEpochs_top_x_samples(start_epoch= start_epoch, end_epoch=end_epoch, top_percent=top_percent)
        return result

    def get_history(self):
        return self.sorted_hist_batchLoss



    def sample_loss(self, index=0):
        epoch_sorted_batchLoss = self.sorted_hist_batchLoss[self.current_epoch]
        loss = epoch_sorted_batchLoss[index]
        return loss

    def sample_loss_range_epoch(self, index=0, start_epoch=0, end_epoch=0):
        range_loss = [d.get(index) for i, d in enumerate(self.sorted_hist_batchLoss) if start_epoch <= i <= end_epoch]
        return range_loss

    def topX_epochs_loss(self, index=0, top_percent=None):
        if top_percent == None:
            top = self.top_percent
        else:
            top = top_percent

        temp_results = list()
        for epoch_no in range(self.epoch):
            temp_result = self.get_top_x_sample(top_percent=top, which_epoch=epoch_no)
            temp_results.append(temp_result)

        result = get_top_epochs(index, temp_results)
        return len(result), result

    def topX_range_epochs_loss(self, index=0, top_percent=None, start_epoch=0, end_epoch=0):
        if top_percent == None:
            top = self.top_percent
        else:
            top = top_percent

        temp_results = list()
        for epoch in range(start_epoch, end_epoch + 1):
            temp_result = self.get_top_x_sample(top_percent=top, which_epoch=epoch)
            temp_results.append(temp_result)

        result = get_top_epochs(index, temp_results, start_epoch)
        return len(result), result

    def save_loss_values_excel(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.sorted_hist_batchLoss, file, indent=4)

