import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
#https://github.com/UKPLab/sentence-transformers
from sentence_transformers import SentenceTransformer

class SemanticStressMatcher:
    def __init__(self, data_path="SAD_v1.xlsx", model_name="all-MiniLM-L6-v2", cache_path="embedding_cache.pkl", sentences = None, severities = None, labels = None):
        self.isEval = False
        self.model_name = model_name
        # control performance by cpu bounding
        self.model = SentenceTransformer(model_name, device='cpu')
        self.result = {}
        # check if running on eval set
        if sentences and severities and labels:
            self.sentences = sentences
            self.severity = severities
            self.labels = labels
            self.isEval = True

        if self.isEval:
            print("Eval Dataset")
            self.embeddings = self.model.encode(self.sentences, convert_to_tensor=False)
        elif os.path.exists(cache_path):
            print("Loading cached embeddings...")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
                self.sentences = cache["sentences"]
                self.embeddings = cache["embeddings"]
                self.labels = cache["labels"]
                self.severity = cache["severity"]
        elif os.path.exists(data_path):
            print("Generating embeddings...")
            df = pd.read_excel(data_path)
            self.sentences = df['sentence'].fillna('').tolist()
            self.labels = df['is_stressor'].fillna(0).astype(int).tolist()
            self.severity = df['avg_severity'].fillna(0).tolist()
            self.embeddings = self.model.encode(self.sentences, convert_to_tensor=False)

            with open(cache_path, "wb") as f:
                pickle.dump({
                    "sentences": self.sentences,
                    "embeddings": self.embeddings,
                    "labels": self.labels,
                    "severity": self.severity
                }, f)
            print("Embeddings cached.")

    # find the closes vector to the user input (gets closest)
    def find_closest(self, user_input, top_k=1):
        #input sentence into a high-dimensional vector using the transformer (return as numpy array not PyTorch)
        input_embedding = self.model.encode([user_input], convert_to_tensor=False)
        #closest vectors but as one value (cosine) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        similarities = cosine_similarity(input_embedding, self.embeddings)[0]
        

        # Get the largest value (most similar)
        best_index = 0
        best_score = similarities[0]
        for i in range(1, len(similarities)):
            if similarities[i] > best_score:
                best_score = similarities[i]
                best_index = i

        self.result ={
            "match": self.sentences[best_index],
            "similarity": float(similarities[best_index]),
            "is_stressor": int(self.labels[best_index]),
            "avg_severity": float(self.severity[best_index])
        }
        print(self.result)
        return self.result


    def threshold_serverity(self):
        avgsvr = self.result['avg_severity']
        if(avgsvr >= 3):
            return "high_stress"
        elif (avgsvr < 3):
            return "moderate_stress"
        elif (avgsvr < 1 or not self.result['is_stressor']):
            return "low_stress"
