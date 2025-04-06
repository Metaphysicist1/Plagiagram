import os
import glob
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MODEL_LOAD_TIMEOUT = 300  # 5 minutes timeout

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

def load_model_with_timeout():
    """Load the model with a timeout and GPU support."""
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model in a separate thread with timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                SentenceTransformer,
                EMBEDDING_MODEL,
                device=device
            )
            
            try:
                model = future.result(timeout=MODEL_LOAD_TIMEOUT)
                logger.info(f"Successfully loaded model on {device}")
                return model
            except TimeoutError:
                logger.error("Model loading timed out")
                raise
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def get_embedding(text, file_path, model):
    """Get embedding for text using Hugging Face model."""
    try:
        # Get file extension and language
        ext = os.path.splitext(file_path)[1]
        language = CODE_EXTENSIONS.get(ext, "Unknown")
        
        # Truncate text if it's too long
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Create context for embedding
        context = f"Programming language: {language}\n\n{text}"
        
        # Generate embedding
        embedding = model.encode(context)
        
        # Ensure we have a numpy array of floats
        embedding = embedding.astype(np.float32)
        
        # Convert to list of floats
        embedding_list = embedding.tolist()
        
        # Verify all values are floats
        if not all(isinstance(x, float) for x in embedding_list):
            logger.error("Could not convert all values to floats")
            return None
            
        return embedding_list
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return None

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        indexes = pc.list_indexes()
        index_exists = PINECONE_INDEX_NAME in [index.name for index in indexes.indexes]
        
        # Create index if it doesn't exist
        if not index_exists:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine"
            )
            logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Get the index
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("Successfully connected to Pinecone")
        
        # Check existing vectors
        stats = index.describe_index_stats()
        logger.info(f"Found {stats.total_vector_count} existing vectors in Pinecone index")
        
        return index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def get_file_paths():
    """Get all code file paths from the repositories."""
    file_paths = []
    
    for ext in CODE_EXTENSIONS.keys():
        pattern = os.path.join(REPOS_STORAGE_PATH, "**", f"*{ext}")
        file_paths.extend(glob.glob(pattern, recursive=True))
    
    logger.info(f"Found {len(file_paths)} code files to process")
    return file_paths

def read_file_content(file_path):
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        return None

def test_embedding(model):
    """Test the embedding function with a simple example."""
    test_code = """
def hello_world():
    print("Hello, World!")
    """
    
    logger.info("Testing embedding function...")
    embedding = get_embedding(test_code, "test.py", model)
    
    if embedding is not None:
        logger.info(f"Embedding type: {type(embedding)}")
        logger.info(f"Embedding length: {len(embedding)}")
        logger.info(f"First 5 values: {embedding[:5]}")
        
        # Test if it's a list of floats
        is_list_of_floats = isinstance(embedding, list) and all(isinstance(x, float) for x in embedding)
        logger.info(f"Is list of floats: {is_list_of_floats}")
        
        return is_list_of_floats
    else:
        logger.error("Failed to generate test embedding")
        return False

def process_files_batch(file_paths, index, model):
    """Process a batch of files and upload to Pinecone."""
    batch = []
    successful = 0
    failed = 0
    
    for file_path in file_paths:
        try:
            content = read_file_content(file_path)
            if not content:
                logger.debug(f"Skipping file (empty content): {file_path}")
                failed += 1
                continue
            
            # Get relative path from the repositories directory
            rel_path = os.path.relpath(file_path, REPOS_STORAGE_PATH)
            repo_name = rel_path.split(os.path.sep)[0]
            
            # Get embedding
            embedding = get_embedding(content, file_path, model)
            if not embedding:
                logger.debug(f"Skipping file (no embedding): {file_path}")
                failed += 1
                continue
            
            # Create vector record
            vector_id = f"{repo_name}_{rel_path}".replace("/", "_").replace("\\", "_")
            batch.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "repo": repo_name,
                    "path": rel_path,
                    "language": CODE_EXTENSIONS.get(os.path.splitext(file_path)[1], "Unknown"),
                    "content": content[:500]  # Store first 500 chars for context
                }
            })
            successful += 1
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            failed += 1
    
    # Upload batch to Pinecone
    if batch:
        try:
            index.upsert(vectors=batch)
            logger.info(f"Uploaded {len(batch)} vectors to Pinecone (successful: {successful}, failed: {failed})")
        except Exception as e:
            logger.error(f"Failed to upload batch to Pinecone: {str(e)}")
    else:
        logger.warning("No vectors to upload in this batch")

def main():
    """Main function to vectorize code repositories."""
    try:
        # Initialize Pinecone
        index = initialize_pinecone()
        
        # Load model with timeout
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = load_model_with_timeout()
        
        # Test embedding function first
        if not test_embedding(model):
            logger.error("Embedding test failed. Exiting.")
            return
        
        # Get all file paths
        file_paths = get_file_paths()
        
        # Process files in batches
        total_batches = (len(file_paths) - 1) // BATCH_SIZE + 1
        for i in range(0, len(file_paths), BATCH_SIZE):
            batch_paths = file_paths[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{total_batches} ({len(batch_paths)} files)")
            process_files_batch(batch_paths, index, model)
            
        logger.info("Finished processing all files")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 