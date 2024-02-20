import torch
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {"state": item[0], "action": item[1]}
    
    def getDimensions(self):
        states = []
        actions = []
        for d in self.data:
            states.append(d[0])
            actions.append(d[1])
        
        states = np.array(states)
        actions = np.array(actions)
        return len(states), states.shape, actions.shape, np.max(states, axis=0, keepdims=True), np.min(states, axis=0, keepdims=True), np.max(actions,axis=0,keepdims=True), np.min(actions,axis=0,keepdims=True
                                                                                                                                                                                    )
