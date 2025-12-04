import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Cargamos el dataset limpio

DATA_PATH = "data/processed/dataset_canciones.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} songs.")


# Gemma 2 

MODEL_NAME = "google/gemma-2-2b"

print("Loading Gemma model (this may take a minute)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# Se crean los embeddings

def embed_text(text):
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**tokens)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze().numpy()

# Se generan los embeddings

embeddings = []

print("Generating embeddings for all lyrics...")
for i, lyric in enumerate(df["Lyric"]):

    print(f"  {i+1}/{len(df)}", end="\r")

    vec = embed_text(lyric)
    embeddings.append(vec)

embeddings = np.vstack(embeddings)
print("\nDone! Embeddings shape:", embeddings.shape)


# Guardamos los embeddings

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/gemma_embeddings.npy", embeddings)
print("Embeddings saved to data/processed/gemma_embeddings.npy")
