import os
import glob
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
REPOS_STORAGE_PATH = "/data/repositories"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
BATCH_SIZE = 100
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024

# File extensions to process
CODE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".go": "Go",
    ".html": "HTML",
    ".css": "CSS",
    ".php": "PHP",
    ".rb": "Ruby",
    ".ts": "TypeScript"
}

def get_embedding(text, file_path, model):
    """Get embedding for text using model."""
    try:
        ext = os.path.splitext(file_path)[1]
        language = CODE_EXTENSIONS.get(ext, "Unknown")
        
        # Truncate text if it's too long
        if len(text) > 8000:
            text = text[:8000]
        
        context = f"Programming language: {language}\n\n{text}"
        embedding = model.encode(context)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return None

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        indexes = pc.list_indexes()
        if PINECONE_INDEX_NAME in [index.name for index in indexes.indexes]:
            logger.info(f"Deleting existing index: {PINECONE_INDEX_NAME}")
            pc.delete_index(PINECONE_INDEX_NAME)
        
        # Create new index with serverless spec
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=spec
        )
        
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def process_files_batch(file_paths, index, model):
    """Process a batch of files and upload to Pinecone."""
    batch = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content:
                continue
            
            rel_path = os.path.relpath(file_path, REPOS_STORAGE_PATH)
            repo_name = rel_path.split(os.path.sep)[0]
            
            embedding = get_embedding(content, file_path, model)
            if not embedding:
                continue
            
            vector_id = f"{repo_name}_{rel_path}".replace("/", "_").replace("\\", "_")
            batch.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "repo": repo_name,
                    "path": rel_path,
                    "language": CODE_EXTENSIONS.get(os.path.splitext(file_path)[1], "Unknown"),
                    "content": content[:500]
                }
            })
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
    
    if batch:
        try:
            index.upsert(vectors=batch)
            logger.info(f"Uploaded {len(batch)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Failed to upload batch to Pinecone: {str(e)}")

def main():
    """Main function to vectorize code repositories."""
    try:
        index = initialize_pinecone()
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        file_paths = []
        for ext in CODE_EXTENSIONS.keys():
            pattern = os.path.join(REPOS_STORAGE_PATH, "**", f"*{ext}")
            file_paths.extend(glob.glob(pattern, recursive=True))
        
        logger.info(f"Found {len(file_paths)} code files to process")
        
        for i in range(0, len(file_paths), BATCH_SIZE):
            process_files_batch(file_paths[i:i+BATCH_SIZE], index, model)
            
        logger.info("Finished processing all files")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()