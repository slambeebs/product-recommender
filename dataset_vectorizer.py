import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
df = pd.read_csv("clean_dataset.csv")

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode product descriptions
descriptions = df["Description"].astype(str).tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)

# Save embeddings back to the DataFrame
df["embedding"] = embeddings.tolist()

# Save to file for later use
df.to_pickle("vectorized_dataset.pkl")
