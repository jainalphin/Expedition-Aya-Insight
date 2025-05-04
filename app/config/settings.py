import os
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file
load_dotenv()

# Configure logging for the httpx library to suppress warnings
logging.getLogger("httpx").setLevel(logging.WARNING)

# Define default paths for documents and summaries output
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "samples/pdf5")
SUMMARIES_OUTPUT_DIR = os.getenv("SUMMARIES_OUTPUT_DIR", "summaries")

# Create the output directory if it doesn't exist
os.makedirs(SUMMARIES_OUTPUT_DIR, exist_ok=True)

# Define models for embedding, reranking, and language model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-v4.0")
COHERERANK_MODEL = os.getenv('COHERERANK_MODEL', 'rerank-v3.5')
LLM_MODEL = os.getenv("LLM_MODEL", "command-a-03-2025")

# Define settings for text splitting and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
COHERERANK_TOPN = int(os.getenv("COHERERANK_TOPN", "100"))
VECTOSTORE_TOPK = int(os.getenv("VECTOSTORE_TOPK", "100"))