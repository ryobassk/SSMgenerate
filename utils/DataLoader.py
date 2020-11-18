import numpy as np
from sklearn.utils import shuffle
import torch
from torch.nn.utils.rnn import pad_sequence
np.random.seed(123)
def sort(x, t):
        lens = [len(i) for i in x]        
        indices = sorted(range(len(lens)), key=lambda i: -lens[i])      
        x = [x[i] for i in indices]
        t = [t[i] for i in indices]
        return (x, t)

class DataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 batch_first=False,
                 device='cpu',
                 random_state=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_first = batch_first
        self.device = device
        if random_state is None:
            random_state = np.random.RandomState(123)
        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()
        x, t = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
        (x, t) = sort(x, t)
        x1,x2,x3=[],[],[]
        x1_sub,x2_sub,x3_sub=[],[],[]
        for i in range(len(x)):
            for k in range(len(x[i])):
                x1_sub.append(x[i][k][0])
                x2_sub.append(x[i][k][1])
                x3_sub.append(x[i][k][2])
            x1.append(x1_sub)
            x2.append(x2_sub)
            x3.append(x3_sub)
            x1_sub,x2_sub,x3_sub=[],[],[]
        t1,t2,t3=[],[],[]
        t1_sub,t2_sub,t3_sub=[],[],[]
        t_chord,t_num =[],[]
        for i in range(len(t)):
            for k in range(len(t[i][0])):
                t1_sub.append(t[i][0][k][0])
                t2_sub.append(t[i][0][k][1])
                t3_sub.append(t[i][0][k][2])
            t1.append(t1_sub)
            t2.append(t2_sub)
            t3.append(t3_sub)
            t1_sub,t2_sub,t3_sub=[],[],[]
            t_chord.append(t[i][1])
            t_num.append(t[i][2])
        
        x1 = [torch.LongTensor(X) for X in x1]
        x2 = [torch.LongTensor(X) for X in x2]
        x3 = [torch.LongTensor(X) for X in x3]
        
        t1 = [torch.LongTensor(T) for T in t1]
        t2 = [torch.LongTensor(T) for T in t2]
        t3 = [torch.LongTensor(T) for T in t3]
        
        t_chord = [torch.LongTensor(T) for T in t_chord]
        t_num = [torch.LongTensor(T) for T in t_num]
        
        x1 = pad_sequence(x1)
        t1 = pad_sequence(t1)
        x2 = pad_sequence(x2)
        t2 = pad_sequence(t2)
        x3 = pad_sequence(x3)
        t3 = pad_sequence(t3)
        t_chord = pad_sequence(t_chord)
        t_num = pad_sequence(t_num)
        
        if self.batch_first:
            x1 = x1.t()
            t1 = t1.t()
            x2 = x2.t()
            t2 = t2.t()
            x3 = x3.t()
            t3 = t3.t()
            t_chord = t_chord.t()
            t_num = t_num.t()
            
        self._idx += self.batch_size
        return (x1.to(self.device), x2.to(self.device), x3.to(self.device),
                t_chord.to(self.device), t_num.to(self.device),
                t1.to(self.device), t2.to(self.device), t3.to(self.device))

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,
                                   random_state=self.random_state)
        self._idx = 0
