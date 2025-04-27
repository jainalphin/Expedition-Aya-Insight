"""
document_processor.py - Text extraction from documents

A submodule for extracting text from PDF documents and images
that can be easily integrated into other projects.
"""

import os
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# For PDFs
from pypdf import PdfReader

# For images
import pytesseract
from PIL import Image

# Configure logging with a null handler by default
# so parent applications can configure as needed
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def process_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text as a string
    """
    try:
        logger.debug(f"Processing PDF: {file_path}")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        return ""


def process_image(file_path: str, lang: str = "eng") -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        file_path: Path to the image file
        lang: OCR language(s) to use

    Returns:
        Extracted text as a string
    """
    try:
        logger.debug(f"Processing image: {file_path}")
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing image {file_path}: {e}")
        return ""


def process_file(file_path: str, lang: str = "eng") -> Dict[str, Any]:
    """
    Process a single file based on its extension.

    Args:
        file_path: Path to the file
        lang: OCR language(s) to use for image processing

    Returns:
        Dictionary containing processing results and metadata
    """
    file_ext = Path(file_path).suffix.lower()

    result = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "text": "",
        "error": None,
        "type": None
    }

    try:
        if file_ext == '.pdf':
            result["text"] = process_pdf(file_path)
            result["type"] = "pdf"
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            result["text"] = process_image(file_path, lang)
            result["type"] = "image"
        else:
            result["error"] = f"Unsupported file type: {file_ext}"
    except Exception as e:
        result["error"] = str(e)

    return result


def process_files(file_paths: List[str], lang: str = "eng", max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process multiple files in parallel.

    Args:
        file_paths: List of file paths to process
        lang: OCR language(s) to use for image processing
        max_workers: Maximum number of worker processes (defaults to CPU count)

    Returns:
        List of dictionaries with processing results
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    logger.info(f"Processing {len(file_paths)} files with {max_workers} workers")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, lang): file_path
            for file_path in file_paths
        }

        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                file_path = future_to_file[future]
                logger.error(f"Exception processing {file_path}: {e}")
                results.append({
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "text": "",
                    "error": str(e),
                    "type": None
                })

    return results


def get_supported_files(folder_path: str, recursive: bool = True) -> List[str]:
    """
    Get all supported files in a folder.

    Args:
        folder_path: Path to the folder
        recursive: Whether to include subfolders

    Returns:
        List of file paths
    """
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    file_paths = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file).suffix.lower() in supported_extensions:
                    file_paths.append(file_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in supported_extensions:
                file_paths.append(file_path)

    return file_paths


def process_folder(folder_path: str, lang: str = "eng", recursive: bool = True, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process all supported files in a folder.

    Args:
        folder_path: Path to the folder containing documents
        lang: OCR language(s) to use for image processing
        recursive: Whether to process subfolders recursively
        max_workers: Maximum number of worker processes

    Returns:
        List of dictionaries with processing results
    """
    file_paths = get_supported_files(folder_path, recursive)
    logger.info(f"Found {len(file_paths)} supported files in {folder_path}")

    return process_files(file_paths, lang, max_workers)


def save_text_files(results: List[Dict[str, Any]], output_dir: str = "extracted_texts") -> Dict[str, int]:
    """
    Save extracted text to files.

    Args:
        results: List of processing results
        output_dir: Directory to save the text files

    Returns:
        Dictionary with counts of successful and failed saves
    """
    os.makedirs(output_dir, exist_ok=True)

    stats = {"success": 0, "skipped": 0, "failed": 0}

    for result in results:
        if not result["text"]:
            stats["skipped"] += 1
            continue

        try:
            output_path = os.path.join(output_dir, f"{result['file_name']}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            stats["success"] += 1
        except Exception as e:
            logger.error(f"Error saving text from {result['file_path']}: {e}")
            stats["failed"] += 1

    return stats


# Example of how this can be used
if __name__ == "__main__":
    # This is just for testing the module
    import time

    # Configure basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    test_folder = "./pdfs"
    start_time = time.time()

    results = process_folder(test_folder, lang="eng+fra+hin", recursive=True)

    print(f"Processed {len(results)} files in {time.time() - start_time:.2f} seconds")

    # Count successes and failures
    success_count = sum(1 for r in results if r["text"] and not r["error"])
    error_count = sum(1 for r in results if r["error"])

    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")

    # Save the results
    stats = save_text_files(results, "extracted_texts")
    print(f"Text files saved: {stats['success']}")