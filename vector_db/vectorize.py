import os
import glob
import logging
from pathlib import Path
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, embed_content
import time
from pinecone import Pinecone  # Updated import

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPOS_STORAGE_PATH = "/data/repositories"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BATCH_SIZE = 100

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

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists with retry logic."""
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Initialize Pinecone with the new API
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists
            indexes = pc.list_indexes()
            index_exists = PINECONE_INDEX_NAME in [index.name for index in indexes.indexes]
            
            # Create index if it doesn't exist
            if not index_exists:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=768,  # Dimension for Gemini embeddings
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
            
            # Get the index
            index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Successfully connected to Pinecone")
            return index
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded. Could not connect to Pinecone.")
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
    """Read content from a file with proper encoding handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {str(e)}")
            return None

def get_embedding(text, file_path):
    """Get embedding for text using Gemini API."""
    try:
        # Get file extension and language
        ext = os.path.splitext(file_path)[1]
        language = CODE_EXTENSIONS.get(ext, "Unknown")
        
        # Create context for embedding
        context = f"Programming language: {language}\n\n{text}"
        
        # Generate embedding
        embedding_result = embed_content(
            model="models/embedding-001",
            content=context,
            task_type="retrieval_document"
        )
        
        # Make sure we're getting a list, not a method
        if hasattr(embedding_result, 'values') and callable(embedding_result.values):
            return embedding_result.values()  # Call the method if it's callable
        elif hasattr(embedding_result, 'values'):
            return embedding_result.values  # Use the property if it's not callable
        else:
            return embedding_result  # Return the result directly if no values attribute
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return None

def process_files_batch(file_paths, index):
    """Process a batch of files and upload to Pinecone."""
    batch = []
    
    for file_path in file_paths:
        content = read_file_content(file_path)
        if not content:
            continue
            
        # Get relative path from the repositories directory
        rel_path = os.path.relpath(file_path, REPOS_STORAGE_PATH)
        repo_name = rel_path.split(os.path.sep)[0]
        
        # Get embedding
        embedding = get_embedding(content, file_path)
        if not embedding:
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
                "content": content[:1000]  # Store first 1000 chars for context
            }
        })
    
    # Upload batch to Pinecone
    if batch:
        index.upsert(vectors=batch)
        logger.info(f"Uploaded {len(batch)} vectors to Pinecone")

def main():
    """Main function to vectorize code repositories."""
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Get all file paths
    file_paths = get_file_paths()
    
    # Process files in batches
    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_paths = file_paths[i:i+BATCH_SIZE]
        process_files_batch(batch_paths, index)
        logger.info(f"Processed batch {i//BATCH_SIZE + 1}/{(len(file_paths)-1)//BATCH_SIZE + 1}")

if __name__ == "__main__":
    main() 