import torch
from torch.utils.data import Dataset


def create_entry(x,y):
    entry = {'past_price': x, 'target': y}
    return entry



class GetDataset(Dataset):
    def __init__(self, args, data):
        super().__init__()

        self.entries = []

        for i in range(args.back_days,len(data)):
            self.entries.append(create_entry(data[i-args.back_days:i],data[i]))




    def __getitem__(self, index):
        entry = self.entries[index]
        input = entry['past_price']
        target = entry['target']
        return input,target

    def __len__(self):
        return len(self.entries)