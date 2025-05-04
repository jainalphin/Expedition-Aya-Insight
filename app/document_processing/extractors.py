"""
Document extraction functionality for processing documents.
"""
import json
import os
import concurrent.futures
import time

import cohere
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from ..config.settings import CHUNK_SIZE, LLM_MODEL

# Configure logging with a null handler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DocumentProcessor:
    """Base class for document processors"""

    def __init__(self):
        self.supported_extensions = []

    def can_process(self, file_path: str) -> bool:
        """Check if the processor can handle this file type"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    def process(self, file_path: str, **kwargs) -> str:
        """Process the document and extract text"""
        raise NotImplementedError("Subclasses must implement this method")


class PdfProcessor(DocumentProcessor):
    """Processor for PDF documents"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']

    def process(self, file_path: str, **kwargs) -> str:
        """Extract text from a PDF file"""
        try:
            # Import here to avoid dependency if not used
            from pypdf import PdfReader

            logger.debug(f"Processing PDF: {file_path}")
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise


class ImageProcessor(DocumentProcessor):
    """Processor for image files"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        # Default languages including multiple options
        self.default_languages = "eng+fra+hin+spa+chi-sim"

    def process(self, file_path: str, **kwargs) -> str:
        """Extract text from an image file using OCR"""
        try:
            # Import here to avoid dependency if not used
            import pytesseract
            from PIL import Image

            # Use the expanded default languages if not specified
            lang = kwargs.get('lang', self.default_languages)
            logger.debug(f"Processing image: {file_path} with languages: {lang}")
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            raise


class DocumentExtractor:
    """Main class for document text extraction"""

    def __init__(self):
        """Initialize with default processors"""
        self.processors = [
            PdfProcessor(),
            ImageProcessor()
        ]
        self.cohere_client = None

    def add_processor(self, processor: DocumentProcessor) -> None:
        """Add a custom document processor"""
        self.processors.append(processor)

    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """Get the appropriate processor for a file"""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    def get_language(self, text: str) -> str:
        """
        Detect the language of the provided text using Cohere API.

        Args:
            text: Text sample to analyze

        Returns:
            String containing the detected language name
        """
        try:
            # Initialize client if not already done
            start = time.time()
            if not self.cohere_client:
                self.cohere_client = cohere.Client()

            prompt = f"What language is this sentence written in?\n\n{text}\n\nRespond only with the language name."
            response = self.cohere_client.chat(
                model=LLM_MODEL,
                message= prompt,
                max_tokens=100,
                temperature=0.2,
            )
            return response.text

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "unknown"

    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single file based on its extension.

        Args:
            file_path: Path to the file
            **kwargs: Additional processing options

        Returns:
            Dictionary containing processing results and metadata
        """
        result = {
            "file_path": file_path,
            "filename": Path(file_path).name,
            "text": "",
            "error": None,
            "type": None,
            "language": None,
            "chunk_size": 0
        }

        try:
            processor = self.get_processor(file_path)

            if processor:
                text = processor.process(file_path, **kwargs)
                result["text"] = text
                result["language"] = self.get_language(text[:CHUNK_SIZE]) if text else None
                result["type"] = processor.__class__.__name__.lower().replace('processor', '')
            else:
                ext = Path(file_path).suffix.lower()
                result["error"] = f"Unsupported file type: {ext}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def process_files(self, file_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            **kwargs: Additional processing options
                     (max_workers: max number of processes)

        Returns:
            List of dictionaries with processing results
        """
        max_workers = kwargs.pop('max_workers', os.cpu_count() or 1)
        logger.info(f"Processing {len(file_paths)} files with {max_workers} workers")

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers * 2) as executor:
            futures = {
                executor.submit(self.process_file, file_path, **kwargs): file_path
                for file_path in file_paths
            }

            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {e}")
                    results.append({
                        "filepath": file_path,
                        "filename": Path(file_path).name,
                        "text": "",
                        "error": str(e),
                        "type": None,
                        "langugae": None,
                        "chunk_size": 0
                    })

        return results

    def find_supported_files(self, folder_path: str, recursive: bool = True) -> List[str]:
        """
        Get all supported files in a folder.

        Args:
            folder_path: Path to the folder
            recursive: Whether to include subfolders

        Returns:
            List of file paths
        """
        # Get all supported extensions from processors
        supported_extensions = []
        for processor in self.processors:
            supported_extensions.extend(processor.supported_extensions)

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

    def process_folder(self, folder_path: str, recursive: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        Process all supported files in a folder.

        Args:
            folder_path: Path to the folder containing documents
            recursive: Whether to process subfolders recursively
            **kwargs: Additional processing options

        Returns:
            List of dictionaries with processing results
        """
        file_paths = self.find_supported_files(folder_path, recursive)
        logger.info(f"Found {len(file_paths)} supported files in {folder_path}")

        return self.process_files(file_paths, **kwargs)


class FileOutputManager:
    """Class for managing output of extracted text"""

    def __init__(self, output_dir: str = "extracted_texts"):
        """Initialize with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Save extracted text to files.

        Args:
            results: List of processing results

        Returns:
            Dictionary with counts of successful and failed saves
        """
        stats = {"success": 0, "skipped": 0, "failed": 0}

        for result in results:
            if not result["text"]:
                stats["skipped"] += 1
                continue

            try:
                # Create filename with original name + file type
                base_name = Path(result['filename']).stem
                file_type = result.get('type', 'unknown')
                output_filename = f"{base_name}_{file_type}.txt"

                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                stats["success"] += 1
            except Exception as e:
                logger.error(f"Error saving text from {result['file_path']}: {e}")
                stats["failed"] += 1

        return stats


# Adapter class to convert DocumentExtractor results to langchain Document objects
class DocumentProcessorAdapter:
    """
    Adapter to process documents and convert them to langchain Document objects.
    """
    def __init__(self):
        """Initialize document processor adapter with the extractor."""
        self.extractor = DocumentExtractor()

    def process_folder(self, folder_path):
        """
        Process all documents in a folder.

        Args:
            folder_path (str): Path to the folder containing documents

        Returns:
            tuple: (list of langchain Document objects, original extraction results)
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Extract content from documents
        extraction_results = self.extractor.process_folder(folder_path)
        print(f"Processed {len(extraction_results)} documents")
        return extraction_results