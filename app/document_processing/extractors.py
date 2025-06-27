"""
Document extraction functionality for processing documents and URLs.
"""
import json
import os
import concurrent.futures
import time
import requests
from urllib.parse import urlparse
from io import BytesIO

import cohere
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langsmith.wrappers import wrap_openai
from openai import OpenAI

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
    """Enhanced processor for PDF documents with multiple extraction methods"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']

    def process(self, file_path, **kwargs) -> str:
        """Extract text from a PDF file using multiple methods for better extraction"""
        try:
            # Try PyPDF2 first (more reliable for most PDFs)
            try:
                import PyPDF2
                logger.debug(f"Processing PDF with PyPDF2: {file_path}")

                if isinstance(file_path, str):
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                else:
                    # Handle BytesIO object (from URL downloads)
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"

                if text.strip():
                    return text.strip()
            except ImportError:
                logger.debug("PyPDF2 not available, trying pypdf")

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    def can_process(self, file_path: str) -> bool:
        if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.supported_extensions:
            return True
        return False


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

            if isinstance(file_path, str):
                image = Image.open(file_path)
            else:
                image = Image.open(file_path)

            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            raise


class UrlProcessor(DocumentProcessor):
    """Processor for web URLs"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = []  # URLs don't have extensions in the traditional sense
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def can_process(self, file_path: str) -> bool:
        """Check if this is a valid URL"""
        try:
            result = urlparse(file_path)
            return all([result.scheme, result.netloc])  # Must have scheme (http/https) and netloc (domain)
        except:
            return False

    def process(self, file_path: str, **kwargs) -> str:
        """Process URL content based on content type"""
        try:
            logger.debug(f"Processing URL: {file_path}")

            # Set timeout and other request parameters
            timeout = kwargs.get('timeout', 30)

            response = self.session.get(file_path, timeout=timeout, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()

            if 'application/pdf' in content_type:
                return self._process_pdf_from_url(response)
            elif any(img_type in content_type for img_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff']):
                return self._process_image_from_url(response, **kwargs)
            elif 'text/html' in content_type or 'text/plain' in content_type:
                return self._process_html_from_url(response)
            else:
                # Try to process as text anyway
                text_content = response.text
                return text_content.strip()

        except Exception as e:
            logger.error(f"Error processing URL {file_path}: {e}")
            raise

    def _process_pdf_from_url(self, response):
        """Process PDF content from URL response"""
        try:
            pdf_content = BytesIO(response.content)
            pdf_processor = PdfProcessor()
            return pdf_processor.process(pdf_content)
        except Exception as e:
            logger.error(f"Error processing PDF from URL: {e}")
            raise

    def _process_image_from_url(self, response, **kwargs):
        """Process image content from URL response"""
        try:
            image_content = BytesIO(response.content)
            image_processor = ImageProcessor()
            return image_processor.process(image_content, **kwargs)
        except Exception as e:
            logger.error(f"Error processing image from URL: {e}")
            raise

    def _process_html_from_url(self, response):
        """Process HTML content from URL response"""
        try:
            # Try to use BeautifulSoup for better HTML parsing
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                return text
            except ImportError:
                logger.warning("BeautifulSoup not available, using basic HTML processing")
                # Basic HTML processing without BeautifulSoup
                import re
                html_content = response.text
                # Remove script and style content
                html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                html_content = re.sub(r'<[^>]+>', '', html_content)
                # Clean up whitespace
                html_content = re.sub(r'\s+', ' ', html_content)
                return html_content.strip()

        except Exception as e:
            logger.error(f"Error processing HTML from URL: {e}")
            raise


class DocumentExtractor:
    """Main class for document text extraction"""

    def __init__(self):
        """Initialize with default processors"""
        self.processors = [
            PdfProcessor(),
            UrlProcessor()
        ]
        self.openai_client = None

    def add_processor(self, processor: DocumentProcessor) -> None:
        """Add a custom document processor"""
        self.processors.append(processor)

    def get_processor(self, file_path: str) -> Optional[DocumentProcessor]:
        """Get the appropriate processor for a file or URL"""
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
            if not self.openai_client:
                self.openai_client = wrap_openai(OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY")))

            prompt = f"What language is this sentence written in?\n\n{text}\n\nRespond only with the language name."
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2,
            )
            lang = response.choices[0].message.content.strip()
            return lang

        except Exception as e:
            print(e)
            raise e
            logger.error(f"Error detecting language: {e}")
            return "unknown"

    def process_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process a single file or URL based on its type.

        Args:
            file_path: Path to the file or URL
            **kwargs: Additional processing options

        Returns:
            Dictionary containing processing results and metadata
        """
        result = {
            "filename": file_path,
            "text": "",
            "error": None,
            "type": None,
            "language": None,
            "chunk_size": 0,
            "is_url": self._is_url(file_path)
        }

        try:
            processor = self.get_processor(file_path)
            print("processor", processor)

            if processor:
                text = processor.process(file_path, **kwargs)
                result["text"] = text
                result["language"] = self.get_language(text[:CHUNK_SIZE]) if text else None
                result["type"] = processor.__class__.__name__.lower().replace('processor', '')
                result["chunk_size"] = len(text)
            else:
                if self._is_url(file_path):
                    result["error"] = f"Unsupported URL content type or unable to process URL"
                else:
                    ext = Path(file_path).suffix.lower()
                    result["error"] = f"Unsupported file type: {ext}"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _is_url(self, path: str) -> bool:
        """Check if the path is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False

    def process_files(self, file_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple files/URLs in parallel.

        Args:
            file_paths: List of file paths or URLs to process
            **kwargs: Additional processing options
                     (max_workers: max number of processes)

        Returns:
            List of dictionaries with processing results
        """
        max_workers = kwargs.pop('max_workers', os.cpu_count() or 1)
        logger.info(f"Processing {len(file_paths)} files/URLs with {max_workers} workers")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                        "filename": file_path,
                        "text": "",
                        "error": str(e),
                        "type": None,
                        "language": None,
                        "chunk_size": 0,
                        "is_url": self._is_url(file_path)
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
        # Get all supported extensions from processors (excluding URL processor)
        supported_extensions = []
        for processor in self.processors:
            if not isinstance(processor, UrlProcessor):
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
                if result.get("is_url", False):
                    # For URLs, use domain name
                    base_name = urlparse(result['file_path']).netloc.replace('.', '_')
                else:
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

    def process_items(self, items: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of files or URLs.

        Args:
            items: List of file paths or URLs

        Returns:
            List of extraction results
        """
        extraction_results = self.extractor.process_files(items)
        print(f"Processed {len(extraction_results)} items")
        return extraction_results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("../../../../.env")
    extractor = DocumentExtractor()
    items = [
        "https://arxiv.org/pdf/1706.03762",
        "https://aclanthology.org/D19-3019.pdf"
    ]
    results = extractor.process_files(items)
    print(results)