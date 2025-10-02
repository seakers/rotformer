import pickle
import os

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.entity_idxs = {self.entities[i]:i for i in range(len(self.entities))}
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.relation_idxs = {self.relations[i]:i for i in range(len(self.relations))}
        self.train_contextual_triplets = self.get_contextual_triplets(data_dir, "train", self.train_data)
        self.valid_contextual_triplets = self.get_contextual_triplets(data_dir, "valid", self.valid_data)
        self.test_contextual_triplets = self.get_contextual_triplets(data_dir, "test", self.test_data)

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
    
    def get_contextual_triplets(self, data_dir, data_type, data):
        contextual = {}
        try:
            print(f"{data_dir}{data_type}_contextual_triplets.pk")
            with open(f"{data_dir}{data_type}_contextual_triplets.pk", "rb") as file:
                contextual = pickle.load(file)
                print(len(contextual))
        except FileNotFoundError as error:
            entities = self.get_entities(data)
            for e in entities:
                contextual.update({self.entity_idxs[e]: [[self.entity_idxs[d[0]],self.relation_idxs[d[1]],self.entity_idxs[d[2]]] for d in data if e in d]})
            with open(f"{data_dir}{data_type}_contextual_triplets.pk", "wb") as file:
                pickle.dump(contextual, file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            print(type(error),error)
        return contextual

def main():
    data = Data(data_dir="data/WN18RR/", reverse=True)

if __name__ == "__main__":
    main()
