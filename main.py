import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config.settings import DOCS_FOLDER
from app.document_processing.extractors import DocumentProcessorAdapter
from app.retrieval.vector_store import Retriever
from app.summarization.summarizer import DocumentSummarizer
from app.summarization.output import SummaryOutputManager


def process_documents() -> List[Dict[str, Any]]:
    """Extract and preprocess documents from the configured folder."""
    processor = DocumentProcessorAdapter()
    return processor.process_folder(DOCS_FOLDER)


def setup_retrieval_system(
    extraction_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Retriever]:
    """Initialize the retriever and attach chunk sizes to documents."""
    retriever = Retriever()
    updated_results = retriever.create_from_documents(extraction_results)
    return updated_results, retriever


def batch_summarize_documents(
    extraction_results: List[Dict[str, Any]],
    summarizer: DocumentSummarizer,
    output_manager: SummaryOutputManager
) -> List[Dict[str, Any]]:
    """Summarize all documents in parallel using a thread pool."""
    results = []

    def process_file(result: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        filename = result.get("filename")
        try:
            summary = summarizer.summarize_document(
                filename=filename,
                language=result.get("language"),
                chunk_size=result.get("chunk_size")
            )
            output_paths = output_manager.save_summary(filename, summary, formats=["markdown"])
            return {
                "filename": filename,
                "success": True,
                "output_paths": output_paths,
                "processing_time": time.time() - start_time,
            }
        except Exception as e:
            return {
                "filename": filename,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, res): res["filename"] for res in extraction_results}
        for future in as_completed(futures):
            results.append(future.result())

    return results


def main():
    start_time = time.time()
    print(f"[INFO] Starting document processing from: {DOCS_FOLDER}")

    extraction_results = process_documents()
    print(f"[INFO] Document processing completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    extraction_results, retriever = setup_retrieval_system(extraction_results)
    print(f"[INFO] Retriever setup completed in {time.time() - start_time:.2f} seconds.")

    summarizer = DocumentSummarizer(retriever)
    output_manager = SummaryOutputManager(output_dir="summaries")

    print("[INFO] Starting batch summarization...")
    start_time = time.time()
    results = batch_summarize_documents(extraction_results, summarizer, output_manager)

    print("\n[SUMMARY] Document Summarization Results:")
    for result in results:
        if result["success"]:
            print(f"✓ {result['filename']} ({result['processing_time']:.2f}s)")
        else:
            print(f"✗ {result['filename']}: {result.get('error')}")

    successful_count = sum(1 for r in results if r["success"])
    print("\n" + "=" * 50)
    print(f"Summarized {successful_count}/{len(results)} documents")
    print(f"Total summarization time: {time.time() - start_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
