import os
from typing import List, Optional
from pathlib import Path
from embeddings import CodeEmbeddingStore
from llm import CodeLLM

class CodeRAG:
    """
    Retrieval-Augmented Generation system for code understanding and assistance.
    Combines code embeddings for retrieval with an LLM for generating responses.
    Uses ChromaDB for vector storage.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2", 
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system with embedding store and LLM
        
        Args:
            api_key: API key for the LLM (defaults to GEMINI_API env variable if None)
            model_name: Name of the embedding model to use
            persist_directory: Directory to persist ChromaDB
        """
        self.embedding_store = CodeEmbeddingStore(model_name=model_name, 
                                                 persist_directory=persist_directory)
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.getenv("GEMINI_API")
            if api_key is None:
                raise ValueError("No API key provided and GEMINI_API environment variable not set")
        
        self.llm = CodeLLM(api_key=api_key)
    
    def index_codebase(self, directory_path: str, file_extensions: List[str] = ['.py']):
        """
        Index all code files in a directory with specified extensions
        
        Args:
            directory_path: Path to the directory containing code files
            file_extensions: List of file extensions to include
        """
        code_files = []
        for ext in file_extensions:
            code_files.extend([str(p) for p in Path(directory_path).glob(f"**/*{ext}")])
        
        print(f"Indexing {len(code_files)} files...")
        self.embedding_store.add_code_files(code_files)
        print("Indexing complete!")
    
    def check_index(self) -> bool:
        """Check if the ChromaDB collection has documents"""
        stats = self.embedding_store.get_collection_stats()
        return stats["count"] > 0
    
    def query(self, user_query: str, k: int = 5) -> str:
        """
        Process a user query about the codebase
        
        Args:
            user_query: The user's question about the code
            k: Number of relevant code chunks to retrieve
            
        Returns:
            Generated response based on retrieved context
        """
        if not self.check_index():
            return "Please index your codebase first using index_codebase()"
        
        # Retrieve relevant code chunks
        relevant_chunks = self.embedding_store.query(user_query, k=k)
        
        # Format chunks for the LLM
        context = []
        for chunk in relevant_chunks:
            source = chunk["metadata"]["source"]
            text = chunk["text"]
            context.append(f"File: {source}\n{text}")
        
        # Generate response using LLM with context
        response = self.llm.generate_response(user_query, context)
        
        return response
    
    def inspect_database(self):
        """Print information about the vector database"""
        stats = self.embedding_store.get_collection_stats()
        print(f"ChromaDB Collection Stats:")
        print(f"- Document count: {stats['count']}")
        print(f"- Embedding model: {stats['model']}")
        print(f"- Storage location: {stats['persist_directory']}")
        
        # If there are documents, show a sample query
        if stats["count"] > 0:
            print("\nSample query results for 'function definition':")
            results = self.embedding_store.query("function definition", k=2)
            for i, result in enumerate(results):
                source = result["metadata"]["source"]
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                print(f"\nResult {i+1} from {source}:")
                print(text_preview)

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = CodeRAG()
    
    # Either index a new codebase
    # rag.index_codebase("./my_project")
    
    # Or check existing index
    if rag.check_index():
        print("Using existing code index")
        rag.inspect_database()
    else:
        print("No index found. Please index your codebase first.")
    
    # Query the system
    response = rag.query('''
                        class Operation():
                            def __init__(self,x):
                                return x+=x
                         
                         ''')
    print("\nQuery Response:")
    print(response)
