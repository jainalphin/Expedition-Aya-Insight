import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import cohere
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config.settings import LLM_MODEL


class DocumentSummarizer:
    """
    Processes and summarizes documents by extracting relevant content
    for different components using vector search and reranking.
    """

    def __init__(self, retriever, batch_size: int = 4):
        """
        Initialize the document summarizer.

        Args:
            retriever: Vector retriever object for finding relevant document chunks
            batch_size: Number of concurrent summarization operations
        """
        self.batch_size = batch_size
        self.retriever = retriever
        self.cohere_client = cohere.ClientV2()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Define component names and their descriptive titles
        self.components = {
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

        # Initialize prompts for each component
        self.prompts = self._initialize_prompts()

        # Validate that prompts and components match
        self._validate_components_and_prompts()

        # Define the order of sections in the final summary
        self.sections_order = [
            'basic_info', 'abstract',
            'methods', 'results', 'equations', 'technical',
            'related_work', 'applications', 'limitations'
        ]

    def _validate_components_and_prompts(self) -> None:
        """
        Validate that all components have corresponding prompts and vice versa.
        Logs warnings for any mismatches.
        """
        missing_prompts = [comp for comp in self.components if comp not in self.prompts]
        if missing_prompts:
            self.logger.warning(f"No prompts found for components: {missing_prompts}")

        missing_components = [prompt_key for prompt_key in self.prompts if prompt_key not in self.components]
        if missing_components:
            self.logger.warning(f"Prompts found for components not in self.components: {missing_components}")

    def _initialize_prompts(self) -> Dict[str, str]:
        """
        Initialize prompts for each component from the prompt module.

        Returns:
            Dictionary mapping component keys to their respective prompts
        """
        try:
            from ..summarization.prompts import (
                basic_info_prompt, abstract_prompt,
                methods_prompt, results_prompt, limitations_prompt,
                related_work_prompt, applications_prompt,
                technical_prompt, equations_prompt,
                # These are imported but not used in the component mapping:
                visuals_prompt, contributions_prompt,
                quick_summary_prompt, reading_guide_prompt
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
            self.logger.error(f"Error importing prompts: {e}")
            return {}

    def summarize_text(self, documents: List[str], prompt: str, language: str) -> Optional[str]:
        """
        Summarizes the provided documents using the given prompt and language
        via the Cohere Chat API.

        Args:
            documents: List of document texts to summarize
            prompt: The prompt to guide the summarization
            language: Target language for the summary

        Returns:
            Summarized text or None if the API call fails
        """
        if not documents:
            self.logger.warning("No documents provided for summarization.")
            return None

        # Format documents for Cohere API
        formatted_docs = [{"text": doc} for doc in documents]

        try:
            response = self.cohere_client.chat(
                model=LLM_MODEL,
                documents=formatted_docs,
                messages=[
                    {"role": "system", "content": f"You are an expert summarization AI. Please respond in {language}."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract text from response, with proper error handling
            if (response and
                    hasattr(response, 'message') and
                    response.message and
                    hasattr(response.message, 'content') and
                    response.message.content and
                    len(response.message.content) > 0 and
                    hasattr(response.message.content[0], 'text')):
                return response.message.content[0].text
            else:
                self.logger.warning(f"Unexpected API response structure for prompt: {prompt[:50]}...")
                return None

        except Exception as e:
            self.logger.error(f"Error during Cohere API call: {str(e)}")
            return None

    def extract_relevant_documents(self, component: str, filename: str, chunk_size: int) -> List[str]:
        """
        Extracts relevant document chunks for a specific component.

        Args:
            component: The component key to extract documents for
            filename: The filename to filter documents by
            chunk_size: The number of chunks in the document

        Returns:
            List of relevant document texts
        """
        component_description = self.components.get(component, component)
        query = f"Analyze the {component_description} section from the document titled '{filename}'."

        try:
            documents = self.retriever.get_relevant_docs(
                chromdb_query=query,
                rerank_query=query,
                filter={'filename': filename},
                chunk_size=chunk_size
            )
            return documents
        except Exception as e:
            self.logger.error(f"Error during document retrieval for component {component}: {str(e)}")
            return []

    def summarize_document(self, filename: str, language: str, chunk_size: int) -> str:
        """
        Summarizes a document by processing each component in parallel.

        Args:
            filename: The name of the document to summarize
            language: Target language for the summary
            chunk_size: The number of chunks in the document

        Returns:
            Compiled summary of the document
        """
        start_total = time.time()
        components = list(self.components.keys())
        results = {}
        errors = {}

        self.logger.info(f"Starting summarization for document: {filename}")

        def process_component(comp: str) -> Tuple[str, Optional[str], Optional[str]]:
            """Process a single component and return results/errors"""
            comp_start = time.time()
            self.logger.info(f"Starting processing for component: {comp}")

            try:
                document_chunks = self.extract_relevant_documents(comp, filename, chunk_size)

                if not document_chunks:
                    self.logger.warning(f"No documents found for component: {comp} in file {filename}")
                    return comp, None, "No relevant documents found"

                prompt = self.prompts.get(comp)
                if not prompt:
                    self.logger.warning(f"No prompt defined for component: {comp}")
                    return comp, None, "No prompt defined"

                # Summarize the retrieved documents
                summary = self.summarize_text(document_chunks, prompt, language)

                comp_end = time.time()
                self.logger.info(f"Finished processing component: {comp}. Time: {comp_end - comp_start:.2f}s")
                return comp, summary, None

            except Exception as e:
                comp_end = time.time()
                error_msg = str(e)
                self.logger.error(f"Error processing component {comp}: {error_msg}. Time: {comp_end - comp_start:.2f}s")
                return comp, None, error_msg

        # Determine optimal worker count based on batch size
        max_workers = min(len(components), self.batch_size)

        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_component = {executor.submit(process_component, comp): comp for comp in components}

            # Process results as they complete
            for future in as_completed(future_to_component):
                comp = future_to_component[future]
                try:
                    comp_name, result, error = future.result()
                    if result is not None:
                        results[comp_name] = result
                    elif error:
                        errors[comp_name] = error
                except Exception as exc:
                    self.logger.error(f"{comp} generated an exception during result retrieval: {str(exc)}")
                    errors[comp] = str(exc)

        end_total = time.time()
        total_time = end_total - start_total
        success_count = len(results)
        error_count = len(errors)

        self.logger.info(
            f"Completed summarization of {filename} in {total_time:.2f}s. "
            f"Successful components: {success_count}, Failed components: {error_count}"
        )

        if errors:
            self.logger.warning(f"Components with errors: {list(errors.keys())}")

        # Compile the available results
        compiled = self.compile_summary(filename, results)
        return compiled

    def compile_summary(self, filename: str, results: Dict[str, str]) -> str:
        """
        Compiles a summary by concatenating the results of all components.

        Args:
            filename: The name of the document being summarized
            results: Dictionary mapping component keys to their summary text

        Returns:
            Compiled summary text in Markdown format
        """
        generation_time = time.strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            f"# Summary of {filename}",
            f"Generated on: {generation_time}\n"
        ]

        # Add sections in the predefined order
        for section in self.sections_order:
            if section in results and results[section]:
                title = self.components.get(section, section).title()
                lines.append(f"## {title}\n")
                lines.append(f"{results[section]}\n")

        # Add any sections that are in results but not in sections_order
        for section, content in results.items():
            if section not in self.sections_order and content:
                title = self.components.get(section, section).title()
                lines.append(f"## {title}\n")
                lines.append(f"{content}\n")

        return "\n".join(lines)