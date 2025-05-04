import logging  # Import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from app.config.settings import DOCS_FOLDER
# Import classes from the renamed modules
from app.document_processing.extractors import DocumentProcessorAdapter
from app.retrieval.vector_store import Retriever
from app.summarization.output import SummaryOutputManager
from app.summarization.summarizer import DocumentSummarizer

# Configure logging for the main script
logger = logging.getLogger(__name__)



def process_uploaded_files(uploaded_files) -> List[Dict[str, Any]]:
    """
    Processes a list of files uploaded via Streamlit.
    Saves them temporarily into a folder and uses the DocumentProcessorAdapter
    to process that folder.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
                        Type hint is omitted here to avoid needing Streamlit import at top level.

    Returns:
        List of dictionaries with original extraction results, including chunk_size.
    """
    # Import streamlit here, as it's used for st.warning
    import streamlit as st

    start_time = time.time()
    logger.info(f"Starting processing for {len(uploaded_files)} uploaded files.")

    # Create a temporary directory to save uploaded files
    # This directory will be automatically cleaned up when the 'with' block exits
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Using temporary directory: {tmpdir}")
        # Save all uploaded files into the temporary directory
        for uploaded_file in uploaded_files:
            # Create a safe path within the temporary directory
            # Use uploaded_file.name directly, tempfile handles uniqueness if needed
            file_path = os.path.join(tmpdir, uploaded_file.name)
            # Write the file content to the temporary path
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                logger.debug(f"Saved uploaded file '{uploaded_file.name}' to '{file_path}'")
            except Exception as e:
                logger.error(f"Error saving uploaded file '{uploaded_file.name}' to temporary directory: {e}", exc_info=True)
                # Log a warning in Streamlit if a file couldn't be saved
                st.warning(f"Could not save uploaded file '{uploaded_file.name}' temporarily. It will be skipped.")


        # Use the DocumentProcessorAdapter to process the entire temporary folder
        processor = DocumentProcessorAdapter() # Corrected typo here
        # Call process_folder with the temporary directory path
        extraction_results = processor.process_folder(tmpdir)
        # The process_folder method returns the list of extraction results

    end_time = time.time()
    logger.info(f"Finished processing uploaded files in {end_time - start_time:.2f} seconds.")
    # The extraction_results list now contains dictionaries with 'filename', 'text', 'error', etc.
    return extraction_results


def setup_retrieval_system(extraction_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Retriever]:
    """
    Sets up the retrieval system (vector store) from extraction results.

    Args:
        extraction_results: List of dictionaries from document extraction.
                            Should contain 'filename' and 'text'.

    Returns:
        A tuple containing:
        - The updated extraction_results list (with 'chunk_size' populated by Retriever).
        - An initialized Retriever instance.
    """
    start_time = time.time()
    logger.info("Setting up retrieval system.")
    try:
        retriever = Retriever()
        # create_from_documents takes extraction results, chunks text, embeds, and builds the DB.
        # It also updates the extraction_results list with the 'chunk_size' for each document.
        updated_extraction_results = retriever.create_from_documents(extraction_results)
        end_time = time.time()
        logger.info(f"Retriever setup complete in {end_time - start_time:.2f} seconds.")
        return updated_extraction_results, retriever
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error during retrieval system setup: {e}", exc_info=True)
        # If retrieval setup fails, the summarization cannot proceed.
        # Re-raise the exception so the Streamlit app can catch and display it.
        raise


def summarize_extracted_documents(extraction_results: List[Dict[str, Any]], retriever: Retriever) -> List[Dict[str, Any]]:
    """
    Summarizes documents based on extraction results and a configured retriever.

    Args:
        extraction_results: List of dictionaries from document extraction (should include chunk_size
                            populated by setup_retrieval_system).
        retriever: An initialized Retriever instance.

    Returns:
        A list of dictionaries, each containing the summary result for a file.
        Each dictionary includes:
        - 'filename': The name of the file.
        - 'success': Boolean indicating if summarization was successful.
        - 'summary': The generated summary string (if successful), or None.
        - 'error': An error message string (if not successful), or None.
        - 'processing_time': Time taken for summarization of this file.
    """
    start_time = time.time()
    logger.info(f"Starting summarization for {len(extraction_results)} documents.")

    # Initialize the summarizer with the retriever
    summarizer = DocumentSummarizer(retriever)

    results = [] # List to store results for each document

    # Filter out results that failed extraction or have no text/chunks
    # Summarization requires extracted text and successful chunking (chunk_size > 0)
    summarizable_results = [
        res for res in extraction_results
        if res.get('text') and res.get('chunk_size', 0) > 0 and res.get('error') is None
    ]
    skipped_results = [
        res for res in extraction_results
        if res not in summarizable_results
    ]

    if skipped_results:
        logger.warning(f"Skipping summarization for {len(skipped_results)} files due to extraction errors or no text/chunks.")
        for res in skipped_results:
            # Add entries for skipped files to the results list
            results.append({
                'filename': res.get('filename', 'unknown'),
                'success': False,
                'summary': None,
                'error': res.get('error', 'Extraction failed or no text/chunks'),
                'processing_time': 0, # No summarization time for skipped files
            })


    def process_single_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function to summarize a single document result."""
        file_start_time = time.time()
        filename = result.get('filename', 'unknown')
        # Use detected language, default to English if detection failed
        language = result.get('language', 'en')
        chunk_size = result.get('chunk_size', 0) # Should be > 0 for summarizable_results

        logger.info(f"Summarizing document: {filename}")

        try:
            # Call the summarizer for a single document
            # The summerize_document method handles parallel processing of components internally
            summary = summarizer.summerize_document(filename, language, chunk_size)

            file_end_time = time.time()
            logger.info(f"Finished summarizing {filename} in {file_end_time - file_start_time:.2f} seconds.")
            return {
                'filename': filename,
                'success': True,
                'summary': summary, # Return the summary string
                'error': None,
                'processing_time': file_end_time - file_start_time,
            }
        except Exception as e:
            file_end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error summarizing document {filename}: {e}", exc_info=True)
            return {
                'filename': filename,
                'success': False,
                'summary': None,
                'error': error_msg,
                'processing_time': file_end_time - file_start_time,
            }

    with ThreadPoolExecutor(max_workers=None) as executor: # Adjust max_workers as needed
        # Submit summarizable document results to the executor
        futures = {executor.submit(process_single_summary, res): res['filename'] for res in summarizable_results}

        # Process results as they complete
        for future in as_completed(futures):
            filename = futures[future]
            try:
                summary_result = future.result()
                results.append(summary_result)
                logger.debug(f"Summary result received for {filename}")
            except Exception as exc:
                # This catches exceptions *within* the future's result retrieval
                logger.error(f"Exception retrieving summary result for {filename}: {exc}", exc_info=True)
                results.append({
                    'filename': filename,
                    'success': False,
                    'summary': None,
                    'error': f"Failed to retrieve result: {exc}",
                    'processing_time': 0, # Can't determine processing time if result retrieval failed
                })

    end_time = time.time()
    logger.info(f"Finished batch summarization in {end_time - start_time:.2f} seconds.")
    return results


# if __name__ == "__main__":
#     start_time = time.time()
#     logger.info("Starting document summarization process (command line).")
#
#     try:
#         # Step 1: Process documents from the predefined folder
#         logger.info(f"Processing documents from: {DOCS_FOLDER}")
#         # DocumentProcessorAdapter().process_folder returns a list of extraction result dicts
#         extraction_results = DocumentProcessorAdapter().process_folder(DOCS_FOLDER)
#         logger.info(f"Document Processing Time taken: {time.time()-start_time:.2f} seconds")
#
#         # Step 2: Setup retrieval system
#         setup_start_time = time.time()
#         # setup_retrieval_system takes extraction results and returns updated results (with chunk_size) and the retriever
#         extraction_results_with_chunks, retriever = setup_retrieval_system(extraction_results)
#         logger.info(f"Retriever Setup Time taken: {time.time() - setup_start_time:.2f} seconds")
#
#         # Step 3: Summarize the documents
#         summarization_start_time = time.time()
#         # For command line, we might still want to save files locally
#         output_manager = SummaryOutputManager() # Uses default output_dir from settings
#         # summarize_extracted_documents performs the summarization and returns results
#         summary_results = summarize_extracted_documents(extraction_results_with_chunks, retriever)
#
#         # Step 4: Save summaries to files (for command-line only)
#         logger.info("Saving summaries to files.")
#         saved_count = 0
#         for res in summary_results:
#             if res['success'] and res['summary']:
#                 # Use the output_manager to save the summary string
#                 output_manager.save_summary(res['filename'], res['summary'], formats=['markdown'])
#                 saved_count += 1
#         logger.info(f"Saved {saved_count} summaries.")
#
#
#         logger.info(f"Summarization Time taken: {time.time() - summarization_start_time:.2f} seconds")
#
#
#         # Output results summary to console
#         logger.info("\n" + "=" * 50)
#         logger.info("Summarization Process Complete.")
#         logger.info("=" * 50)
#         successful_count = sum(res.get('success', False) for res in summary_results)
#         total_processed = len(summary_results) # Includes skipped files if they were added to results list earlier
#         total_time = time.time() - start_time
#
#         logger.info(f"Total files attempted: {len(extraction_results)}") # Total files found/attempted extraction
#         logger.info(f"Files successfully extracted and summarizable: {len(extraction_results_with_chunks)}") # Files with text and chunks
#         logger.info(f"Files summarized: {successful_count}/{total_processed}")
#         logger.info(f"Total process time: {total_time:.2f} seconds")
#         logger.info("=" * 50)
#
#         # Print individual results status
#         logger.info("\nIndividual File Results:")
#         for result in summary_results:
#             name = result.get('filename', 'unknown')
#             status = "SUCCESS" if result['success'] else "FAILED"
#             time_taken = result.get('processing_time', 0)
#             error_msg = result.get('error', '')
#             logger.info(f"- {name}: {status} ({time_taken:.2f}s) {f'Error: {error_msg}' if error_msg else ''}")
#
#
#     except FileNotFoundError as fnf_error:
#         logger.error(f"Configuration Error: {fnf_error}")
#         print(f"Error: {fnf_error}")
#     except Exception as main_error:
#         logger.error(f"An unexpected error occurred during the main process: {main_error}", exc_info=True)
#         print(f"An unexpected error occurred: {main_error}")