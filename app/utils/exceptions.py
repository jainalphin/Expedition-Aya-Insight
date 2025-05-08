class DocumentProcessingError(Exception):
    """
    Exception raised when there's an error processing a document.

    This is typically used when document parsing, extraction, or transformation fails.
    """

    def __init__(self, message="Error occurred while processing document", document_id=None, details=None):
        self.document_id = document_id
        self.details = details

        # Build a more detailed message if additional info is provided
        if document_id:
            message = f"{message} (document_id: {document_id})"
        if details:
            message = f"{message}: {details}"

        super().__init__(message)


class NoRelevantContentError(Exception):
    """
    Exception raised when no relevant content is found in a document or query.

    This is typically used when a document is processed successfully but doesn't
    contain the expected or required content.
    """

    def __init__(self, message="No relevant content found", document_id=None, query=None):
        self.document_id = document_id
        self.query = query

        # Build a more detailed message if additional info is provided
        if document_id:
            message = f"{message} in document (id: {document_id})"
        if query:
            message = f"{message} for query: '{query}'"

        super().__init__(message)