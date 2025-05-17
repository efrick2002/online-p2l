import random

class ReplayBuffer():
    '''
    ideas: start with 100% new example, gradually decrease to eventually 50% new 50% old? 

    only start replay after N time steps 
    '''
    def __init__(self, limit):
        self._buffer = []
        self._limit = limit
        self._next_idx = 0

    def sample(self, sample_cnt):
        assert sample_cnt <= len(self._buffer)
        return random.sample(self._buffer, sample_cnt)
    
    def add(self, novel_data):
        for seq in novel_data:
            if self._next_idx >= len(self._buffer): #if limit not hit
                self._buffer.append(seq)
            else:
                self._buffer[self._next_idx] = seq #else replace old examples
            self._next_idx = (self._next_idx + 1) % self._limit
            
    def __len__(self):
        return len(self._buffer)

    