import logging
import os
from typing import Dict, List, Any, Iterator, Optional

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from app.config.settings import LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from app.utils.exceptions import DocumentProcessingError, NoRelevantContentError
from app.utils.performance import timeit
from app.summarization.prompts import comprehensive_research_paper_prompt

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Processes documents and generates streaming summaries using vector search
    and LLM-based summarization with OpenAI's streaming API.
    """

    def __init__(self, max_workers: int = 16, batch_size: int = 4):
        """Initialize summarizer with vector retriever and configuration."""
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._initialize_text_splitter()
        self._initialize_openai_client()

    def _initialize_text_splitter(self) -> None:
        """Initialize the text splitter with configured chunk size and overlap."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def _initialize_openai_client(self) -> None:
        """Initialize the OpenAI client with environment configuration."""
        self.openai_client = wrap_openai(OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY")
        ))

    @traceable
    def get_streaming_summary(
            self,
            documents: Optional[List[str]|str],
            prompt: str,
            language: str = "en"
    ) -> Iterator:
        """
        Generate a streaming summary using OpenAI's chat API.

        Args:
            documents: List of document texts to summarize
            prompt: The prompt template to use for summarization
            language: Target language for the summary

        Returns:
            Iterator that yields events as content is generated

        Raises:
            NoRelevantContentError: If no documents are provided
            DocumentProcessingError: If API call fails
        """
        if not documents:
            raise NoRelevantContentError("No document content provided for summarization")

        
        try:
            return self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert summarization AI. Please respond in {language}."
                    },
                    {
                        "role": "user",
                        "content": prompt.format(research_paper=documents)
                    }
                ],
                stream=True
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise DocumentProcessingError(f"Failed to generate summary: {str(e)}")

    def _get_document_chunks(self, document_details: Dict[str, Any]) -> List[str]:
        """
        Retrieve relevant document chunks for a specific component using vector search.

        Args:
            document_details: Dictionary containing document information

        Returns:
            List of document text chunks
        """
        try:
            text = document_details.get('text', '')
            return [text] if text else []
        except Exception as e:
            filename = document_details.get('filename', 'unknown')
            logger.error(f"Document retrieval error for {filename}: {e}")
            return []

    @traceable
    def get_resource_link_stream(self, document_text: str) -> Iterator:
        """
        Generate a streaming response for finding the original research paper link.

        Args:
            document_text: Text content of the document

        Returns:
            Iterator that yields the research paper link

        Raises:
            NoRelevantContentError: If no document text is provided
            DocumentProcessingError: If API call fails
        """
        if not document_text:
            logger.error("Empty document content provided for resource link lookup")
            raise NoRelevantContentError("No document content provided for resource link lookup")

        try:
            truncated_text = document_text[:1000]
            yield self.openai_client.responses.create(
                model=LLM_MODEL,
                tools=[{"type": "web_search_preview", "search_context_size": "low"}],
                input=f"Find the research paper link for this document: {truncated_text} Respond only with the link.",
            ).output_text
        except Exception as e:
            logger.error(f"OpenAI API error in resource link lookup: {e}")
            raise DocumentProcessingError(f"Failed to generate resource link: {str(e)}")

    def _create_error_component(self, filename: str, comp_name: str, error_message: str) -> Dict[str, Any]:
        """Create a component dictionary for error cases."""
        return {
            'filename': filename,
            'comp_name': comp_name,
            'success': False,
            'error': error_message
        }

    def _create_success_component(
            self,
            filename: str,
            comp_name: str,
            stream_generator: Iterator
    ) -> Dict[str, Any]:
        """Create a component dictionary for successful processing."""
        return {
            'filename': filename,
            'comp_name': comp_name,
            comp_name: stream_generator,
            'success': True
        }

    def _process_component(self, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single component for streaming summary generation.

        Args:
            comp_data: Dictionary containing component processing data

        Returns:
            Dictionary containing component information and stream generator
        """
        comp_name = 'summary'
        filename = comp_data['filename']
        language = comp_data.get('language', 'en')
        try:
            # Get relevant document chunks
            document_chunks = self._get_document_chunks(comp_data)

            if not document_chunks:
                logger.warning(f"No relevant content found for {comp_name} in '{filename}'")
                return self._create_error_component(filename, comp_name, 'No relevant content found')

            # Validate prompt availability
            prompt = comprehensive_research_paper_prompt
            if not prompt:
                logger.warning(f"No prompt defined for component: {comp_name}")
                return self._create_error_component(filename, comp_name, 'No prompt defined')

            # Generate streaming summary
            stream_generator = self.get_streaming_summary(document_chunks, prompt, language)
            logger.info(f"Created stream generator for '{filename}' component '{comp_name}'")
            return self._create_success_component(filename, comp_name, stream_generator)

        except Exception as e:
            logger.error(f"Failed to process component '{comp_name}' for '{filename}': {e}")
            return self._create_error_component(filename, comp_name, str(e))