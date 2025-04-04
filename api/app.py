import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import sys
sys.path.append('/app')
from models.plagiarism_detector import PlagiarismDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Initialize plagiarism detector
detector = PlagiarismDetector()

@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    """Endpoint to detect plagiarism in code."""
    try:
        # Get request data
        data = request.json
        
        if not data or 'code' not in data:
            return jsonify({"error": "Missing 'code' field in request"}), 400
        
        code = data['code']
        language = data.get('language', 'Unknown')
        
        # Detect plagiarism
        result = detector.detect_plagiarism(code, language)
        
        # Parse result if it's a string (JSON from Gemini)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                # If parsing fails, return a simplified result
                is_plagiarized = "true" in result.lower() and "is_plagiarized" in result.lower()
                result = {
                    "is_plagiarized": is_plagiarized,
                    "explanation": "Analysis completed, but result format was unexpected."
                }
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/index-stats', methods=['GET'])
def index_stats():
    """Get statistics about the Pinecone index."""
    try:
        # Get the index from the detector
        index = detector.index
        
        # Get stats
        stats = index.describe_index_stats()
        
        return jsonify({
            "total_vector_count": stats.total_vector_count,
            "namespaces": stats.namespaces,
            "dimension": stats.dimension
        }), 200
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}")
        return jsonify({"error": f"Failed to get index stats: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 


    