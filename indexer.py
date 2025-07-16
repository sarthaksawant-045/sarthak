import faiss
import numpy as np
import os
import pickle
from vectorizer import Embedder

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"

embedder = Embedder()

def load_index():
    embed_dim = embedder.model.get_sentence_embedding_dimension()
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(embed_dim)

def save_metadata(meta):
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return []

def index_documents(doc_dict):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    paths = list(doc_dict.keys())
    texts = list(doc_dict.values())

    meta = load_metadata()
    new_paths = [p for p in paths if p not in meta]

    if not new_paths:
        return 0  # No new documents to index

    new_texts = [doc_dict[p] for p in new_paths]

    vectors = embedder.embed_texts(new_texts)
    index = load_index()
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    meta.extend(new_paths)
    save_metadata(meta)

    return len(new_paths)
