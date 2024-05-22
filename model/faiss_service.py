import numpy as np
import faiss

class FaissService:
    def __init__(self, embeddings, ids):
        # print(embeddings.shape, ids.shape)
        if len(embeddings) != len(set(ids)):
            return
        else:
            self.dimension = embeddings.shape[1]
            index_flat = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(index_flat)
            self.index.add_with_ids(embeddings, ids)
            self.data = dict(zip(ids.tolist(), embeddings.tolist()))
        
    def add(self, new_id, new_embedding):
        if new_id in list(self.data.keys()):
            return
        else:
            self.index.add_with_ids(new_embedding, np.array([new_id], dtype='int64'))
            self.data[new_id] = new_embedding.tolist()
        
    def delete(self, id):
        if id not in list(self.data.keys()):
            return
        else:
            self.index.remove_ids(np.array([id], np.int64))
            self.data.pop(id, None)
    
    def update(self, new_id, new_embedding):
        self.delete(new_id)
        self.add(new_id, new_embedding)
        return True

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding, k)
        return D, I