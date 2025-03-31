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
        
        # For plagiarism detection, use a specific prompt format
        if "plagiarism" in user_query.lower():
            prompt = f"""
            You are a plagiarism detection expert. Analyze the following code snippet:
            
            USER CODE:
            {user_query}
            
            POTENTIAL MATCHES FROM DATABASE:
            {'\n\n'.join(context)}
            
            Based on your analysis, is the user code plagiarized from any of the potential matches?
            Important: You must respond with ONLY one word - either "yes" (if plagiarized) or "no" (if not plagiarized).
            """
            return self.llm.generate_response(prompt, [])  # Empty context since we included it in the prompt
        
        # For regular queries, use the standard approach
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
    # Initialize the RAG system
    rag = CodeRAG()  # Make sure you have your API key set as GEMINI_API env variable or pass it directly

    # Index your codebase - replace "./my_project" with the path to your actual code directory
    rag.index_codebase(
        directory_path="./repo_fetching/Repositories_path",  
        file_extensions=['.py', '.js', '.java']  # Add any file extensions you want to index
    )

    # Now you can make queries
    response = rag.query("Your question about the code here")
    print(response)