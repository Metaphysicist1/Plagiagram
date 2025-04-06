import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pinecone_index():
    """Check Pinecone index configuration and stats."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # List all indexes
        indexes = pc.list_indexes()
        logger.info("Available indexes:")
        for idx in indexes.indexes:
            logger.info(f"- {idx.name}")
        
        # Check if our index exists
        if index_name in [idx.name for idx in indexes.indexes]:
            # Get the index
            index = pc.Index(index_name)
            
            # Get index stats
            stats = index.describe_index_stats()
            logger.info(f"\nIndex '{index_name}' stats:")
            logger.info(f"Dimension: {stats.dimension}")
            logger.info(f"Total vectors: {stats.total_vector_count}")
            logger.info(f"Namespaces: {stats.namespaces}")
            
            return stats
        else:
            logger.warning(f"Index '{index_name}' not found!")
            return None
            
    except Exception as e:
        logger.error(f"Error checking Pinecone index: {str(e)}")
        return None

def delete_pinecone_index():
    """Delete the Pinecone index if it exists."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if index exists
        indexes = pc.list_indexes()
        if index_name in [idx.name for idx in indexes.indexes]:
            logger.info(f"Deleting index '{index_name}'...")
            pc.delete_index(index_name)
            logger.info("Index deleted successfully!")
        else:
            logger.warning(f"Index '{index_name}' not found!")
            
    except Exception as e:
        logger.error(f"Error deleting Pinecone index: {str(e)}")

def create_pinecone_index(dimension=384):
    """Create a new Pinecone index with specified dimension."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Check if index exists
        indexes = pc.list_indexes()
        if index_name in [idx.name for idx in indexes.indexes]:
            logger.warning(f"Index '{index_name}' already exists!")
            return False
        
        # Create new index
        logger.info(f"Creating new index '{index_name}' with dimension {dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
        logger.info("Index created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error creating Pinecone index: {str(e)}")
        return False

if __name__ == "__main__":
    # First check current index configuration
    stats = check_pinecone_index()
    
    if stats:
        # Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. Delete the current index")
        print("2. Create a new index with dimension 384")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            delete_pinecone_index()
        elif choice == "2":
            if create_pinecone_index(dimension=384):
                check_pinecone_index()  # Check the new index
        elif choice == "3":
            print("Exiting...")
        else:
            print("Invalid choice!")
    else:
        # If no index exists, ask if user wants to create one
        print("\nNo index found. Would you like to create a new index with dimension 384?")
        choice = input("Enter 'y' to create or 'n' to exit: ")
        
        if choice.lower() == 'y':
            if create_pinecone_index(dimension=384):
                check_pinecone_index()  # Check the new index 