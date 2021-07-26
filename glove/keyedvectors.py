import numpy as np


class KeyedVectors:
    def __init__(self, vector_size, count=0):
        self.vector_size = vector_size
        self.index_to_key = []
        self.key_to_index = {}
        self.vectors = np.zeros((count, vector_size), dtype=np.float32)

    def __len__(self):
        return len(self.index_to_key)

    def __contains__(self, key):
        return key in self.key_to_index

    def __getitem__(self, key):
        return self.vectors[self.key_to_index[key]]

    @classmethod
    def load(cls, filename):
        index_to_key = []
        key_to_index = {}
        vectors = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    index_to_key.append(parts[0])
                    key_to_index[parts[0]] = len(index_to_key) - 1
                    vectors.append(np.array(parts[1:], dtype=np.float32))
        vectors = np.array(vectors, dtype=np.float32)
        wv = cls(vector_size=vectors.shape[1], count=vectors.shape[0])
        wv.vectors = vectors
        wv.index_to_key = index_to_key
        wv.key_to_index = key_to_index
        return wv
