"""
PDF OCR Processing Module

Provides a reusable class for extracting text from PDF files using OCR
via the Marker library with language support.
"""

import logging
import os

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser


class PdfOcrProcessor:
    """
    A class to perform OCR on PDF files and extract text content.

    This class handles the configuration and processing of PDF files
    to extract text using OCR capabilities from the Marker library.
    """

    # Define supported output formats
    SUPPORTED_FORMATS = ["markdown", "html", "json"]

    # Define supported languages with their ISO codes
    # Must be either the names or codes from from https://github.com/VikParuchuri/surya/blob/master/surya/recognition/languages.py.
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ar": "Arabic",
        "zh": "Chinese",
    }

    def __init__(
        self,
        output_format: str = "markdown",
        language: str = "en",
        disable_images: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the PDF OCR processor with the desired configuration.

        Parameters
        ----------
        output_format : str, optional
            Format for the extracted text, by default "markdown".
            Must be one of: "markdown", "html", or "json".
        language : str, optional
            OCR processing, by default "en".
            Check SUPPORTED_LANGUAGES for available options.
        disable_images : bool, optional
            Whether to disable image extraction, by default True
        log_level : int, optional
            Logging level for this processor, by default logging.INFO

        Raises
        ------
        ValueError
            If the output_format is not supported or if language is not supported
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Validate output format
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format: {output_format}. "
                f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            self.logger.warning(
                f"Language '{language}' not in supported list: {list(self.SUPPORTED_LANGUAGES.keys())}. "
                f"Defaulting to English (en)."
            )
            language = "en"

        self.language = language
        self.logger.info(f"OCR language set to: {self.SUPPORTED_LANGUAGES.get(language, 'Unknown')} ({language})")

        # Base configuration
        self.config = {
            "output_format": output_format,
            "disable_image_extraction": disable_images,
            "language": language
        }

        # Initialize configuration parser
        self.config_parser = ConfigParser(self.config)

        # Create the converter
        self.converter = PdfConverter(
            config=self.config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=self.config_parser.get_processors(),
            renderer=self.config_parser.get_renderer(),
            llm_service=self.config_parser.get_llm_service()
        )

        self.logger.debug(f"PdfOcrProcessor initialized with config: {self.config}")

    def process_file(self, pdf_path: str) -> str:
        """
        Process a PDF file and extract text using OCR.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file to process

        Returns
        -------
        str
            Extracted text in the configured format

        Raises
        ------
        FileNotFoundError
            If the PDF file doesn't exist
        ValueError
            If the file isn't a PDF
        RuntimeError
            If OCR processing fails
        """
        # Validate input file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {pdf_path}")

        try:
            self.logger.info(
                f"Running OCR on: {pdf_path} with language: "
                f"{self.SUPPORTED_LANGUAGES.get(self.language, 'Unknown')} ({self.language})"
            )

            # Process the PDF
            rendered = self.converter(pdf_path)

            # Extract text from rendered output
            text, _, _ = text_from_rendered(rendered)

            self.logger.info(
                f"Successfully processed {pdf_path} ({len(text)} characters) "
                f"in {self.SUPPORTED_LANGUAGES.get(self.language, 'Unknown')}"
            )
            return text

        except FileNotFoundError:
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"OCR processing failed: {str(e)}")

    @classmethod
    def get_supported_languages(cls) -> dict:
        """
        Get a dictionary of supported languages.

        Returns
        -------
        dict
            Dictionary mapping language codes to language names
        """
        return cls.SUPPORTED_LANGUAGES


def extract_text_from_pdf(
    pdf_path: str,
    output_format: str = "markdown",
    language: str = "en",
    disable_images: bool = True
) -> str:
    """
    Extract text from a PDF file using OCR (function wrapper for quick use).

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to process
    output_format : str, optional
        Format for the extracted text, by default "markdown".
        Must be one of: "markdown", "html", or "json".
    language : str, optional
        ISO 639-1 language code for OCR processing, by default "en".
        Use PdfOcrProcessor.get_supported_languages() to see available options.
    disable_images : bool, optional
        Whether to disable image extraction, by default True

    Returns
    -------
    str
        Extracted text in the specified format

    Raises
    ------
    FileNotFoundError
        If the PDF file doesn't exist
    ValueError
        If the file isn't a PDF or if the output format is invalid
    RuntimeError
        If OCR processing fails
    """
    processor = PdfOcrProcessor(
        output_format=output_format,
        language=language,
        disable_images=disable_images
    )
    return processor.process_file(pdf_path)


# Example usage
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example 1: Simple function call with English
    try:
        text = extract_text_from_pdf("sample.pdf", language="en")
        print(f"Extracted {len(text)} characters in English")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Processing with different language
    try:
        text = extract_text_from_pdf("sample_spanish.pdf", language="es")
        print(f"Extracted {len(text)} characters in Spanish")
    except Exception as e:
        print(f"Error: {e}")