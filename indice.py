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

print("Loading Gemma model for queries...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed_query(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=600,
        padding=True
    )

    with torch.no_grad():
        output = model(**tokens)

    vec = output.last_hidden_state.mean(dim=1)
    vec = vec.detach().cpu().numpy().astype(np.float32)

    return vec


#Busqueda semántica
def buscar_canciones(query, k=5):
    print("\nQuery:", query)
    q_vector = embed_query(query)

    distances, indices = index.search(q_vector, k)

    print("\nEstas son las 5 canciones más similares:\n")
    for rank, idx in enumerate(indices[0]):
        title = df.iloc[idx]["Title"]
        artist = df.iloc[idx]["Artist"]
        print(f"{rank+1}. {title} — {artist}")

#Probar busqueda 
if __name__ == "__main__":
    while True:
        query = input("\nEscribe tu búsqueda (o 'salir'): ")
        if query.lower() == "salir":
            break
        buscar_canciones(query)
