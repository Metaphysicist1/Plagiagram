import os
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class CodeEmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the embedding store with ChromaDB and a sentence transformer model
        
        Args:
            model_name: Name of the sentence-transformer model to use
            persist_directory: Directory to persist ChromaDB
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence-transformers model for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            embedding_function=self.embedding_function
        )

    def process_code_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Split code file into meaningful chunks with metadata"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
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
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": file_path,
                            "size": len(chunk_text)
                        }
                    })
                    current_chunk = []
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": file_path,
                    "size": len(chunk_text)
                }
            })
            
        return chunks

    def add_code_files(self, file_paths: List[str]):
        """Process and add multiple code files to the store"""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        for file_path in file_paths:
            try:
                chunks = self.process_code_file(file_path)
                for chunk in chunks:
                    all_chunks.append(chunk["text"])
                    all_metadatas.append(chunk["metadata"])
                    all_ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Add chunks in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            self.collection.add(
                documents=all_chunks[i:end],
                metadatas=all_metadatas[i:end],
                ids=all_ids[i:end]
            )
        
        print(f"Added {len(all_chunks)} chunks to ChromaDB")

    def query(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the code store and return most relevant chunks
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
            
        return formatted_results

    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "count": count,
            "model": self.model_name,
            "persist_directory": self.persist_directory
        }
