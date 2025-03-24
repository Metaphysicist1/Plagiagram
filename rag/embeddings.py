import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class CodeEmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding store with a sentence transformer model
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.code_chunks: List[str] = []
        self.faiss_index_path = "code_index.faiss"
        self.chunks_path = "code_chunks.pkl"

    def process_code_file(self, file_path: str) -> List[str]:
        """Split code file into meaningful chunks"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple chunking by function/class definitions and comments
        chunks = []
        current_chunk = []
        
        for line in content.split('\n'):
            current_chunk.append(line)
            
            if (line.startswith('def ') or 
                line.startswith('class ') or 
                line.startswith('"""') or 
                line.strip() == ''):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    def add_code_files(self, file_paths: List[str]):
        """Process and add multiple code files to the store"""
        for file_path in file_paths:
            chunks = self.process_code_file(file_path)
            self.code_chunks.extend(chunks)
        
        # Create embeddings
        embeddings = self.model.encode(self.code_chunks)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index and chunks
        self.save_store()

    def save_store(self):
        """Save the FAISS index and code chunks to disk"""
        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.code_chunks, f)

    def load_store(self):
        """Load the FAISS index and code chunks from disk"""
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.chunks_path, 'rb') as f:
                self.code_chunks = pickle.load(f)

    def query(self, query: str, k: int = 3) -> List[str]:
        """
        Query the code store and return most relevant chunks
        
        Args:
            query: Search query
            k: Number of results to return
        """
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        return [self.code_chunks[i] for i in indices[0]]
