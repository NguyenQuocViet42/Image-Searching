class image_storage:
    def __init__(self, ids_list, embs_list, images_list, boxes_list, list_table):
        self.embs_dict = dict(zip(ids_list, embs_list))
        self.images_dict = dict(zip(ids_list, images_list))
        self.boxes_dict = dict(zip(ids_list, boxes_list))
        self.table_dict = dict(zip(ids_list, list_table))
        
    def add(self, id, emb, image):
        self.embs_dict[id] = emb
        self.images_dict[id] = image
    
    def delete(self, image):
        ids_to_delete = [key for key, val in self.images_dict.items() if val == image]
        for id in ids_to_delete:
            if id in self.embs_dict:
                self.embs_dict.pop(id, None)
            if id in self.images_dict:
                self.images_dict.pop(id, None)