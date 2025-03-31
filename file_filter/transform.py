from filter import filter_files
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

def transform_and_save_to_chroma(collection_name="code_repository"):
    # 1. Get all files
    files = filter_files(
            root_dir="./repo_fetching/Repositories_path/",
            extensions=['.py', '.sh','.js'],
            exclude_dirs=['.git', '__pycache__', 'tests']
            )
    
    # 2. Initialize Chroma client and collection
    chroma_client = chromadb.Client()
    
    # Use a pre-built embedding function (or you can use your own)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get collectionfrom filter import filter_files
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # 3. Process files and add to Chroma
    for i, file in enumerate(files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a unique ID for each document
            doc_id = f"doc_{i}"
            
            # Add to Chroma collection
            collection.add(
                documents=[content],
                metadatas=[{"file_path": file, "file_type": os.path.splitext(file)[1]}],
                ids=[doc_id]
            )
            
            print(f"Added {file} to Chroma database")
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    print(f"Successfully added {len(files)} files to Chroma collection '{collection_name}'")
    return collection

# For chunking larger files (optional but recommended for code files)
def transform_with_chunking(chunk_size=1500, overlap=200, collection_name="code_chunks"):
    # 1. Get all files
    files = filter_files(
            root_dir="./repo_fetching/Repositories_path/",
            extensions=['.py', '.sh','.js'],
            exclude_dirs=['.git', '__pycache__', 'tests']
            )
    
    # 2. Initialize Chroma client and collection
    chroma_client = chromadb.Client()
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # 3. Process and chunk files
    chunk_count = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by lines to preserve code structure
            lines = content.split('\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line_length = len(line)
                if current_length + line_length > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    # Keep some overlap
                    overlap_lines = current_chunk[-int(len(current_chunk) * overlap/chunk_size):]
                    current_chunk = overlap_lines
                    current_length = sum(len(line) for line in current_chunk)
                
                current_chunk.append(line)
                current_length += line_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            # Add chunks to Chroma
            for i, chunk in enumerate(chunks):
                doc_id = f"{os.path.basename(file)}_{i}"
                collection.add(
                    documents=[chunk],
                    metadatas=[{
                        "file_path": file, 
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": os.path.splitext(file)[1]
                    }],
                    ids=[doc_id]
                )
                chunk_count += 1
                
            print(f"Added {len(chunks)} chunks from {file} to Chroma database")
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    print(f"Successfully added {chunk_count} chunks from {len(files)} files to Chroma collection '{collection_name}'")
    return collection

# Example usage
if __name__ == "__main__":
    # Choose one of these approaches:
    
    # Option 1: Store whole files
    collection = transform_and_save_to_chroma()
    
    # Option 2: Store chunked files (better for RAG with large codebases)
    # collection = transform_with_chunking()
    
    # Example query to test the collection
    results = collection.query(
        query_texts=["how to filter files"],
        n_results=2
    )
    print("Sample query results:", results) 