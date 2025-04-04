import os
import pinecone
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel, embed_content

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SIMILARITY_THRESHOLD = 0.85  # Threshold for plagiarism detection

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

class PlagiarismDetector:
    def __init__(self):
        """Initialize the plagiarism detector."""
        # Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Initialize Gemini model
        self.model = GenerativeModel(model_name="gemini-1.5-pro")
        
        logger.info("Plagiarism detector initialized")
    
    def get_embedding(self, code, language="Unknown"):
        """Get embedding for code using Gemini API."""
        try:
            # Create context for embedding
            context = f"Programming language: {language}\n\n{code}"
            
            # Generate embedding
            embedding = embed_content(
                model="models/embedding-001",
                content=context,
                task_type="retrieval_document"
            )
            
            return embedding.values
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None
    
    def search_similar_code(self, embedding, top_k=5):
        """Search for similar code in the vector database."""
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search similar code: {str(e)}")
            return None
    
    def analyze_with_gemini(self, user_code, similar_results):
        """Use Gemini to analyze if the code is plagiarized."""
        if not similar_results or not similar_results.matches:
            return {"is_plagiarized": False, "explanation": "No similar code found in the database."}
        
        # Prepare prompt for Gemini
        prompt = f"""
        I need to determine if the following code is plagiarized from existing repositories.

        USER CODE:
        ```
        {user_code}
        ```

        POTENTIAL MATCHES:
        """
        
        for i, match in enumerate(similar_results.matches[:3]):  # Use top 3 matches
            similarity = match.score
            repo = match.metadata.get("repo", "Unknown")
            path = match.metadata.get("path", "Unknown")
            content = match.metadata.get("content", "")
            
            prompt += f"""
            MATCH {i+1} (Similarity: {similarity:.2f}):
            Repository: {repo}
            Path: {path}
            Code snippet:
            ```
            {content}
            ```
            """
        
        prompt += """
        Based on the code comparison, determine if the user code is plagiarized from any of the matches.
        Consider:
        1. Code structure and logic similarities
        2. Variable/function naming patterns
        3. Comments and documentation
        4. Common programming patterns vs. direct copying
        
        Respond with a JSON object with two fields:
        - "is_plagiarized": true or false
        - "explanation": brief explanation of your decision
        
        Only respond with the JSON object, nothing else.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to analyze with Gemini: {str(e)}")
            return {"is_plagiarized": False, "explanation": f"Error analyzing code: {str(e)}"}
    
    def detect_plagiarism(self, code, language="Unknown"):
        """Detect if the provided code is plagiarized."""
        # Get embedding for the code
        embedding = self.get_embedding(code, language)
        if not embedding:
            return {"is_plagiarized": False, "explanation": "Failed to process code."}
        
        # Search for similar code
        similar_results = self.search_similar_code(embedding)
        if not similar_results:
            return {"is_plagiarized": False, "explanation": "Failed to search for similar code."}
        
        # Check if any result exceeds the similarity threshold
        has_high_similarity = any(match.score > SIMILARITY_THRESHOLD for match in similar_results.matches)
        
        # If high similarity found, analyze with Gemini for detailed check
        if has_high_similarity or len(similar_results.matches) > 0:
            return self.analyze_with_gemini(code, similar_results)
        
        return {"is_plagiarized": False, "explanation": "No significant similarity found."} 