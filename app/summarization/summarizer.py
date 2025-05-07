import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Any, Iterator

import cohere
from cohere import StreamedChatResponseV2

from ..config.settings import LLM_MODEL
from ..utils.exceptions import DocumentProcessingError, NoRelevantContentError
from ..utils.performance import timeit

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    Processes documents and generates streaming summaries using vector search
    and LLM-based summarization with Cohere's streaming API.
    """

    # Define components and their descriptions
    COMPONENT_TYPES = {
        'basic_info': "Basic Paper Information",
        'abstract': "Abstract Summary",
        'methods': "Methodology Summary",
        'results': "Key Results",
        'limitations': "Limitations & Future Work",
        'related_work': "Related Work",
        'applications': "Practical Applications",
        'technical': "Technical Details",
        'equations': "Key Equations",
    }

    # Define the order of sections in the final document
    SECTIONS_ORDER = [
        'basic_info', 'abstract', 'methods', 'results',
        'equations', 'technical', 'related_work',
        'applications', 'limitations'
    ]

    def __init__(self, retriever, max_workers: int = 4, batch_size: int = 4):
        """Initialize summarizer with vector retriever and configuration."""
        self.retriever = retriever
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cohere_client = cohere.ClientV2()
        self._prompts = self._load_prompts()

        # Validate configuration
        self._validate_config()

    @lru_cache(maxsize=1)
    def _load_prompts(self) -> Dict[str, str]:
        """Load and cache prompts for each component type."""
        try:
            from ..summarization.prompts import (
                basic_info_prompt, abstract_prompt,
                methods_prompt, results_prompt, limitations_prompt,
                related_work_prompt, applications_prompt,
                technical_prompt, equations_prompt
            )

            return {
                'basic_info': basic_info_prompt,
                'abstract': abstract_prompt,
                'methods': methods_prompt,
                'results': results_prompt,
                'limitations': limitations_prompt,
                'related_work': related_work_prompt,
                'applications': applications_prompt,
                'technical': technical_prompt,
                'equations': equations_prompt,
            }
        except ImportError as e:
            logger.error(f"Failed to load summarization prompts: {e}")
            return {}

    def _validate_config(self) -> None:
        """Validate that all components have corresponding prompts."""
        if not self._prompts:
            raise ValueError("No prompts loaded for document summarization")

        missing_prompts = [comp for comp in self.COMPONENT_TYPES if comp not in self._prompts]
        if missing_prompts:
            logger.warning(f"Missing prompts for components: {missing_prompts}")

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

    def get_relevant_document_chunks(self, component: str, filename: str, chunk_size: int) -> List[str]:
        """Retrieve relevant document chunks for a specific component using vector search."""
        component_description = self.COMPONENT_TYPES.get(component, component)
        query = f"Analyze the {component_description} section from the document titled '{filename}'."

        try:
            return self.retriever.get_relevant_docs(
                chromdb_query=query,
                rerank_query=query,
                filter={'filename': filename},
                chunk_size=chunk_size
            )
        except Exception as e:
            logger.error(f"Document retrieval error for {component}: {e}")
            return []

    def _process_component(self, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single component for streaming summary generation.
        This is used by the ThreadPoolExecutor to parallelize component processing.
        """
        comp_name = comp_data['comp_name']
        filename = comp_data['filename']
        language = comp_data.get('language', 'en')
        chunk_size = comp_data.get('chunk_size', 1000)

        try:
            # Get relevant document chunks
            document_chunks = self.get_relevant_document_chunks(comp_name, filename, chunk_size)

            if not document_chunks:
                logger.warning(f"No relevant content found for {comp_name} in '{filename}'")
                return {
                    'filename': filename,
                    'comp_name': comp_name,
                    'success': False,
                    'error': 'No relevant content found'
                }

            # Get prompt for this component
            prompt = self._prompts.get(comp_name)
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
            chunk_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Generate streaming summary components for a document using parallel processing.

        Returns a list of component dictionaries, each containing a
        streaming generator for incremental content consumption.
        """
        logger.info(f"Generating summaries for '{filename}' using ThreadPoolExecutor with {self.max_workers} workers")

        # Prepare component data for parallel processing
        component_tasks = [
            {
                'comp_name': comp_name,
                'filename': filename,
                'language': language,
                'chunk_size': chunk_size
            }
            for comp_name in self.COMPONENT_TYPES
        ]

        components = []

        # Process components in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_component, task): task['comp_name']
                for task in component_tasks
            }

            for future in as_completed(futures):
                comp_name = futures[future]
                try:
                    result = future.result()
                    if result['success']:
                        components.append(result)
                except Exception as e:
                    logger.error(f"Thread execution error for '{comp_name}': {e}")

        successful_count = len([c for c in components if c.get('success', False)])
        logger.info(f"Generated {successful_count}/{len(self.COMPONENT_TYPES)} components for '{filename}'")
        return components

    def compile_summary(self, filename: str, results: Dict[str, str]) -> str:
        """Compile a full document summary from component results."""
        generation_time = time.strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            f"# Summary of {filename}",
            f"Generated on: {generation_time}\n"
        ]

        # Add sections in the predefined order
        for section in self.SECTIONS_ORDER:
            if section in results and results[section]:
                title = self.COMPONENT_TYPES.get(section, section).title()
                lines.append(f"## {title}\n")
                lines.append(f"{results[section]}\n")

        # Add any additional sections not in predefined order
        for section, content in results.items():
            if section not in self.SECTIONS_ORDER and content:
                title = self.COMPONENT_TYPES.get(section, section).title()
                lines.append(f"## {title}\n")
                lines.append(f"{content}\n")

        return "\n".join(lines)