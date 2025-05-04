"""
Configuration settings for the document summarization application.
"""
import os
from dotenv import load_dotenv

load_dotenv()
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Default paths
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "/home/lenovo/Alphin/aya/my_contribution/Expedition-Aya-Insight/samples/pdf5")
SUMMARIES_OUTPUT_DIR = os.getenv("SUMMARIES_OUTPUT_DIR", "summaries")


# Embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-v4.0")

# Rank model
COHERERANK_MODEL = os.getenv('COHERERANK_MODEL', 'rerank-v3.5')

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "command-a-03-2025")

# Text splitter settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval settings
COHERERANK_TOPN = 100
VECTOSTORE_TOPK = 100