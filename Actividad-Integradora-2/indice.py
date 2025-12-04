import numpy as np
import pandas as pd
import faiss 
import torch
from transformers import AutoTokenizer, AutoModel

DATA_PATH = "data/processed/dataset_canciones.csv"
EMB_PATH = "data/processed/gemma_embeddings.npy"
MODEL_NAME = "google/gemma-2-2b"

# Cargar dataset y los embeddings
print("Loading dataset and embeddings...")
df = pd.read_csv(DATA_PATH)
embeddings = np.load(EMB_PATH).astype("float32")

print("Embeddings shape:", embeddings.shape)

# Crear índice
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index created with", index.ntotal, "vectors")

