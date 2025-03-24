from typing import List
import google.generativeai as genai
import os

class CodeLLM:
    def __init__(self, api_key: str):
        """Initialize the LLM with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-pro')

    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate response using retrieved context chunks
        
        Args:
            query: User query
            context_chunks: Retrieved code chunks for context
        """
        prompt = self._build_prompt(query, context_chunks)
        
        response = self.model.generate_content(
            [
                {"role": "user", "parts": [prompt]}
            ]
        )
        
        return response.text

    def _build_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Build prompt with query and context"""
        context = "\n\n".join(context_chunks)
        return f"""
        Given the following code context:
        
        {context}
        
        User question: {query}
        
        Please provide a detailed answer based on the code context above.
        """



api_key = os.getenv("GEMINI_API")
llm = CodeLLM(api_key)

data = '''

 def load_store(self):
        """Load the FAISS index and code chunks from disk"""
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.chunks_path, 'rb') as f:
                self.code_chunks = pickle.load(f)

'''
response = llm.generate_response("Check is this code plagiar or not", [data])
print(response) 