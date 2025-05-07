import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

from app.config.settings import DOCS_FOLDER
from app.document_processing.extractors import DocumentProcessorAdapter
from app.retrieval.vector_store import Retriever
from app.summarization.summarizer import DocumentSummarizer
from app.summarization.output import SummaryOutputManager
from app.utils.performance import timeit
from app.utils.logger import setup_logger

logger = logging.getLogger(__name__)


@timeit
def process_documents() -> List[Dict[str, Any]]:
    """Extract and preprocess documents from the configured folder."""
    processor = DocumentProcessorAdapter()
    return processor.process_folder(DOCS_FOLDER)


def setup_retrieval_system(doc_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Retriever]:
    """Initialize the retriever with document data."""
    retriever = Retriever()
    updated_doc_data = retriever.create_from_documents(doc_data)
    return updated_doc_data, retriever


def process_single_document(doc_data: Dict[str, Any], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Process a single document and generate its summary components."""
    try:
        doc_data, retriever = setup_retrieval_system(doc_data)
        # Pass max_workers to DocumentSummarizer
        summarizer = DocumentSummarizer(retriever, max_workers=max_workers)

        components = summarizer.generate_summarizer_components(
            filename=doc_data.get("filename"),
            language=doc_data.get("language", "en"),
            chunk_size=doc_data.get("chunk_size", 1000)
        )
        return components
    except Exception as e:
        logger.error(f"Failed to summarize {doc_data.get('filename')}: {str(e)}")
        return []


@timeit
def batch_summarize_documents(extraction_results: List[Dict[str, Any]],
                              max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Generate summary components with stream generators for all documents in parallel.

    Args:
        extraction_results: List of document data dictionaries
        max_workers: Maximum number of worker threads (defaults to CPU count)

    Returns:
        List of summary component dictionaries with stream generators
    """
    # Determine optimal number of workers if not specified
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 max to avoid API rate limits

    # Calculate workers per document based on total documents
    doc_count = len(extraction_results)
    doc_workers = max(1, min(4, max_workers // max(1, doc_count)))

    logger.info(f"Processing {doc_count} documents with {max_workers} total workers "
                f"({doc_workers} workers per document)")

    summary_component_streams = []

    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit document processing tasks
        futures = {
            executor.submit(process_single_document, doc_data, doc_workers): doc_data.get('filename', 'unknown')
            for doc_data in extraction_results
        }

        # Collect results as they complete
        for future in as_completed(futures):
            doc_name = futures[future]
            try:
                components = future.result()
                if components:
                    summary_component_streams.extend(components)
                    logger.info(f"Successfully processed components for '{doc_name}'")
                else:
                    logger.warning(f"No components generated for '{doc_name}'")
            except Exception as e:
                logger.error(f"Error processing '{doc_name}': {str(e)}")

    logger.info(f"Generated {len(summary_component_streams)} total summary components")
    return summary_component_streams


def consume_stream(stream_data: Dict[str, Any]) -> Tuple[str, str]:
    """Consume a single streaming generator and return the result."""
    file_id = f"{stream_data['filename']}-{stream_data['comp_name']}"
    component_type = stream_data['comp_name']

    try:
        stream_generator = stream_data[component_type]
        content_buffer = []

        logger.info(f"Processing stream for {file_id}")
        print(f"\n{'=' * 50}\nProcessing: {file_id}\n")

        # Process streaming events
        for event in stream_generator:
            if event.type == "content-delta":
                delta_text = event.delta.message.content.text
                content_buffer.append(delta_text)
                print(delta_text, end="", flush=True)

        print(f"\n{'=' * 50}")
        return file_id, "success"
    except Exception as e:
        logger.error(f"Error processing stream {file_id}: {str(e)}")
        return file_id, f"Error: {str(e)}"


@timeit
def process_stream_components(stream_components: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, str]:
    """Process all streaming components in parallel with controlled concurrency."""
    results = {}

    logger.info(f"Processing {len(stream_components)} summary components with {max_workers} workers")

    # Use semaphore pattern for controlled concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(consume_stream, component): component
            for component in stream_components
        }

        # Process results as they complete
        for future in as_completed(futures):
            component_id, status = future.result()
            results[component_id] = status

    return results


def display_summary_results(results: Dict[str, str]) -> None:
    """Display results summary in a clean format."""
    successful = [k for k, v in results.items() if v == 'success']
    failed = [(k, v) for k, v in results.items() if v != 'success']

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(successful)}/{len(results)} components successfully processed")

    if successful:
        print("\nSuccessful components:")
        for comp in successful:
            print(f"  ✓ {comp}")

    if failed:
        print("\nFailed components:")
        for comp, error in failed:
            print(f"  ✗ {comp}: {error}")

    print("=" * 60)


def main():
    """Main execution flow for document processing and summarization."""
    try:
        # Configure logging
        setup_logger()
        logger.info(f"Starting document processing from: {DOCS_FOLDER}")

        # Process documents
        extraction_results = process_documents()
        logger.info(f"Processed {len(extraction_results)} documents")

        # Determine optimal thread counts based on system resources and document count
        cpu_count = os.cpu_count() or 4
        doc_count = len(extraction_results)

        # Calculate optimal workers for summarization
        # More workers for many documents, fewer for few documents
        summary_workers = min(max(2, cpu_count), 8)  # Cap at 8 to avoid API limits

        # Generate summaries with streaming in parallel
        logger.info(f"Starting parallel streaming summarization with {summary_workers} workers")
        stream_components = batch_summarize_documents(
            extraction_results,
            max_workers=summary_workers
        )

        # Process all streams with adaptive concurrency
        # Use fewer workers for consuming streams to avoid overwhelming output
        stream_workers = min(max(2, cpu_count // 2), 4)
        logger.info(f"Processing streams with {stream_workers} workers")
        results = process_stream_components(stream_components, max_workers=stream_workers)

        # Display results
        display_summary_results(results)

    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()