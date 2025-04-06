# Code Plagiarism Detection System

A proof-of-concept system for detecting code plagiarism using vector similarity and LLM analysis. The system processes code repositories, generates embeddings, and uses similarity search to identify potential plagiarism cases.

## Architecture

The system consists of three main components:

1. **Vector Database Service** (`vector_db/`)

   - Processes code repositories
   - Generates embeddings using llama-text-embed-v2 model
   - Stores vectors in Pinecone for similarity search

2. **Plagiarism Detection Service** (`models/`)

   - Uses Pinecone for similarity search
   - Employs Gemini for detailed analysis
   - Generates CSV reports of evaluation results

3. **API Service** (`api/`)
   - FastAPI-based REST API
   - Handles code submission and plagiarism detection requests
   - Serves the web interface

## Prerequisites

- Docker and Docker Compose
- Pinecone API key
- Google Gemini API key
- Python 3.9+

## Environment Variables

Create a `.env` file in the root directory with:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=plagiagram-db
GEMINI_API_KEY=your_gemini_api_key
```

## Project Structure

```
.
├── api/                    # API service
│   ├── app.py             # FastAPI application
│   ├── static/            # Static files for web interface
│   └── Dockerfile         # API service container
├── models/                 # Plagiarism detection service
│   ├── plagiarism_detector.py
│   └── Dockerfile
├── vector_db/             # Vector database service
│   ├── vectorize.py       # Code processing and embedding
│   └── Dockerfile
├── data/                  # Data directory
│   └── repositories/      # Code repositories to process
├── docker-compose.yml     # Docker services configuration
└── README.md
```

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Place code repositories in `data/repositories/`

4. Build and start services:

```bash
docker compose build
docker compose up
```

## Usage

1. **Vector Database Service**

   - Automatically processes code repositories
   - Generates embeddings and stores in Pinecone
   - Runs on container startup

2. **API Service**

   - Access web interface at `http://localhost:5000`
   - Submit code for plagiarism detection
   - View detection results and CSV reports

3. **Plagiarism Detection**
   - Uses vector similarity for initial detection
   - Employs Gemini for detailed analysis
   - Generates CSV reports in `evaluation_results/`

## Demo System Limitations

This is a proof-of-concept system with the following limitations:

1. **Repository Size**

   - Limited to small repositories for demo purposes
   - Maximum file size: 100KB
   - Maximum code length: 8000 characters

2. **Processing Scope**

   - Compares individual code files
   - Does not perform repository-level analysis
   - Limited to specific programming languages

3. **Performance**
   - Processing large repositories requires significant resources
   - Embedding generation is computationally intensive
   - Limited by API rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- Pinecone for vector database
- Google for Gemini model
- NVIDIA for llama-text-embed-v2 model
