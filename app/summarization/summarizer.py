import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Any, Iterator

import cohere
from cohere import StreamedChatResponseV2

from app.config.settings import LLM_MODEL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.exceptions import DocumentProcessingError, NoRelevantContentError
from app.utils.performance import timeit
from langchain_core.documents import Document

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from app.summarization.prompts import comprehensive_research_paper_prompt

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    Processes documents and generates streaming summaries using vector search
    and LLM-based summarization with Cohere's streaming API.
    """
    def __init__(self, retriever, max_workers: int = 16, batch_size: int = 4):
        """Initialize summarizer with vector retriever and configuration."""
        self.retriever = retriever
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.cohere_client = cohere.ClientV2()

    def get_streaming_summary(
            self,
            documents: List[str],
            prompt: str,
            language: str = "en"
    ) -> Iterator[StreamedChatResponseV2]:
        """
        Generate a streaming summary using Cohere's chat API.

        Returns a generator that yields events as content is generated.
        """
        if not documents:
            raise NoRelevantContentError("No document content provided for summarization")

        try:
            return self.cohere_client.chat_stream(
                model=LLM_MODEL,
                documents=documents,
                messages=[
                    {"role": "system", "content": f"You are an expert summarization AI. Please respond in {language}."},
                    {"role": "user", "content": prompt}
                ],
            )
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise DocumentProcessingError(f"Failed to generate summary: {str(e)}")

    def _get_document_chunks(self, document_details) -> List[str]:
        """Retrieve relevant document chunks for a specific component using vector search."""
        try:
            text = document_details['document_text']
            if text:
                return [text]
            return []

        except Exception as e:
            logger.error(f"Document retrieval error for {document_details['filename']}: {e}")
            return []

    def _process_resource_link(self, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process resource link component for streaming generation.
        """
        filename = comp_data['filename']
        document_text = comp_data.get('document_text', '')[:1000]

        try:
            # Generate streaming resource link
            stream_generator = self.get_resource_link_stream(document_text)

            # Create component with stream
            component = {
                'filename': filename,
                'comp_name': 'resource_link',
                'resource_link': stream_generator,
                'success': True
            }

            logger.info(f"Created resource link stream generator for '{filename}'")
            return component

        except Exception as e:
            logger.error(f"Failed to process resource link for '{filename}': {e}")
            return {
                'filename': filename,
                'comp_name': 'resource_link',
                'success': False,
                'error': str(e)
            }

    def get_resource_link_stream(self, document_text: str) -> Iterator:
        """
        Generate a streaming response for finding the original research paper link.
        Returns a generator that yields events as content is generated.
        """
        if not document_text:
            logger.error("Empty document content provided for resource link lookup")
            raise NoRelevantContentError("No document content provided for resource link lookup")

        try:
            cohere_client = cohere.Client()
            yield cohere_client.chat(
                model=LLM_MODEL,
                message=f"Find the research paper link for this document: {document_text[:1000]} Respond only with the link.",
                connectors=[{"id": "web-search"}],
            ).text
        except Exception as e:
            logger.error(f"Cohere API error in resource link lookup: {e}")
            raise DocumentProcessingError(f"Failed to generate resource link: {str(e)}")

    def _process_component(self, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single component for streaming summary generation.
        This is used by the ThreadPoolExecutor to parallelize component processing.
        """
        comp_name = comp_data['comp_name']
        filename = comp_data['filename']
        language = comp_data.get('language', 'en')

        try:
            # Get relevant document chunks
            document_chunks = self._get_document_chunks(comp_data)

            if not document_chunks:
                logger.warning(f"No relevant content found for {comp_name} in '{filename}'")
                return {
                    'filename': filename,
                    'comp_name': comp_name,
                    'success': False,
                    'error': 'No relevant content found'
                }

            # Get prompt for this component
            prompt = comprehensive_research_paper_prompt
            if not prompt:
                logger.warning(f"No prompt defined for component: {comp_name}")
                return {
                    'filename': filename,
                    'comp_name': comp_name,
                    'success': False,
                    'error': 'No prompt defined'
                }

            # Generate streaming summary
            stream_generator = self.get_streaming_summary(document_chunks, prompt, language)

            # Create component with stream
            component = {
                'filename': filename,
                'comp_name': comp_name,
                comp_name: stream_generator,
                'success': True
            }
            logger.info(f"Created stream generator for '{filename}' component '{comp_name}'")
            return component

        except Exception as e:
            logger.error(f"Failed to process component '{comp_name}' for '{filename}': {e}")
            return {
                'filename': filename,
                'comp_name': comp_name,
                'success': False,
                'error': str(e)
            }

    @timeit
    def generate_summarizer_components(
            self,
            filename: str,
            language: str = "en",
            chunk_size: int = 1000,
            document_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate streaming summary components for a document using parallel processing.

        Returns a list of component dictionaries, each containing a
        streaming generator for incremental content consumption.
        """
        logger.info(f"Generating summaries for '{filename}' using ThreadPoolExecutor with {self.max_workers} workers")
        comp_data = {'filename': filename, 'language': language, 'comp_name': 'related_work', 'document_text': document_text, 'chunk_size': chunk_size}
        prompt_gen = self._process_component(comp_data)
        print(prompt_gen)
        components = [prompt_gen]
        components.append(self._process_resource_link(comp_data))
        print(components)
        return components