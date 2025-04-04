import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # For Gemini

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlagiarismDetector:
    def __init__(self):
        """Initialize the plagiarism detector."""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.embedding_model = "all-MiniLM-L6-v2"
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.pinecone_index_name)
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.embedding_model)
        
        # Initialize Gemini for explanations
        genai.configure(api_key=self.gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-pro')
        
        logger.info("Plagiarism detector initialized")
    
    def get_embedding(self, code, language):
        """Get embedding for code."""
        context = f"Programming language: {language}\n\n{code}"
        embedding = self.model.encode(context)
        return embedding.tolist()
    
    def generate_explanation(self, query_code, matched_code, language, similarity_score):
        """Generate a detailed explanation using Gemini."""
        try:
            prompt = f"""
            I need to analyze two code snippets for plagiarism.
            
            QUERY CODE ({language}):
            ```{language}
            {query_code}
            ```
            
            MATCHED CODE ({language}):
            ```{language}
            {matched_code}
            ```
            
            The similarity score between these snippets is {similarity_score:.2f} (on a scale from 0 to 1).
            
            Please provide:
            1. An analysis of whether this is likely plagiarism
            2. Specific similarities between the code snippets
            3. Any notable differences
            4. If it appears to be plagiarism, suggestions for how the code could be modified to be more original
            
            Format your response as a JSON object with the following fields:
            {
                "is_plagiarized": true/false,
                "explanation": "detailed explanation",
                "similarities": ["list", "of", "similarities"],
                "differences": ["list", "of", "differences"],
                "suggestions": ["list", "of", "suggestions"]
            }
            """
            
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return None
    
    def detect_plagiarism(self, code, language):
        """Detect plagiarism in code."""
        try:
            # Get embedding for the code
            embedding = self.get_embedding(code, language)
            
            # Query Pinecone
            results = self.index.query(
                vector=embedding,
                top_k=5,
                include_metadata=True
            )
            
            # Check if there are any matches
            if not results.matches:
                return {
                    "is_plagiarized": False,
                    "explanation": "No similar code found in our database."
                }
            
            # Check similarity scores
            top_match = results.matches[0]
            similarity_score = top_match.score
            
            # Determine if it's plagiarized based on similarity score
            threshold = 0.95  # Adjust this threshold as needed
            is_plagiarized = similarity_score > threshold
            
            # Basic explanation
            if is_plagiarized:
                repo = top_match.metadata.get("repo", "unknown")
                path = top_match.metadata.get("path", "unknown")
                basic_explanation = f"This code is very similar to code found in {repo}/{path} with a similarity score of {similarity_score:.2f}."
            else:
                basic_explanation = f"This code has some similarities to existing code, but the similarity score ({similarity_score:.2f}) is below our plagiarism threshold."
            
            # Get detailed explanation from LLM if similarity is high enough
            detailed_explanation = None
            if similarity_score > 0.8:  # Only use LLM for significant matches
                matched_code = top_match.metadata.get("content", "")
                detailed_explanation = self.generate_explanation(
                    code, matched_code, language, similarity_score
                )
            
            return {
                "is_plagiarized": is_plagiarized,
                "explanation": basic_explanation,
                "detailed_analysis": detailed_explanation,
                "matches": [
                    {
                        "repo": match.metadata.get("repo", "unknown"),
                        "path": match.metadata.get("path", "unknown"),
                        "score": match.score,
                        "snippet": match.metadata.get("content", "")
                    } for match in results.matches
                ]
            }
        except Exception as e:
            logger.error(f"Error detecting plagiarism: {str(e)}")
            return {
                "is_plagiarized": False,
                "explanation": f"Error analyzing code: {str(e)}"
            } 