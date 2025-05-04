"""
Vector database operations for document storage and retrieval.
"""
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, COHERERANK_MODEL, COHERERANK_TOPN, VECTOSTORE_TOPK
import cohere

class Retriever:
    """
    Wrapper for vector database operations.
    """

    def __init__(self, model=EMBEDDING_MODEL):
        self.cohere_client = cohere.Client()
        self.chroma_db = None
        self.embedding_model = CohereEmbeddings(model=model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def create_from_documents(self, extraction_results):
        chunks = []
        for result in extraction_results:
            filename = result['filename']
            text = result['text']
            if text:
                document = Document(
                    page_content=text,
                    metadata={"filename": filename}
                )
                doc_chunks = self.text_splitter.split_documents([document])
                result['chunk_size'] = len(doc_chunks)
                chunks.extend(doc_chunks)

        self.chroma_db = Chroma.from_documents(
            chunks,
            embedding=self.embedding_model
        )
        return extraction_results

    def similarity_search(self, query, k=5, filter=None):
        if not self.chroma_db:
            raise ValueError("Vector store has not been initialized with documents")

        return self.chroma_db.similarity_search(query=query, k=k, filter=filter)

    def reranking(self, query, docs, top_n=10):
        doc_texts = [doc.page_content for doc in docs]
        rerank_response = self.cohere_client.rerank(model=COHERERANK_MODEL, query=query, documents=doc_texts, top_n=top_n)
        # return [docs[result.index] for result in rerank_response.results]
        return [docs[result.index].page_content for result in rerank_response.results]


    def get_relevant_docs(self, chromdb_query, rerank_query, filter, chunk_size):
        dense_topk = min(chunk_size, VECTOSTORE_TOPK)
        reranking_topk = min(chunk_size, COHERERANK_TOPN)
        docs = self.similarity_search(chromdb_query, filter=filter, k=dense_topk)
        if docs:
            return self.reranking(rerank_query, docs, top_n=reranking_topk)
        return []



