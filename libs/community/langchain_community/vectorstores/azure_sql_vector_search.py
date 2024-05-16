from __future__ import annotations

from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from azure_sql_vector_search.classic_vector_search import AzureSQLClassicVectorSearchClient
from azure_sql_vector_search.models import DistanceMetric, VectorSearchClientMode
from azure_sql_vector_search.native_vector_search import AzureSQLNativeVectorSearchClient
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.DOT_PRODUCT


class AzureSQLVectorDB(VectorStore):
    """
    Azure SQL Vector Search

    Uses the classic or native client to insert and query vector data stored in Azure SQL Database
    """

    def __init__(
            self,
            embedding: Embeddings,
            *,
            query_mode: VectorSearchClientMode = VectorSearchClientMode.CLASSIC,
            distance_metric: DistanceMetric = DistanceMetric.DOT_PRODUCT,
            table_prefix: str = "sql_embeddings",
            connection_string: Optional[str] = None,
    ):

        self.embedding = embedding
        self.distance_metric = distance_metric
        self.table_name = table_prefix

        if query_mode == VectorSearchClientMode.CLASSIC:
            self.search_client = AzureSQLClassicVectorSearchClient(connection_string, table_prefix)
        elif query_mode == VectorSearchClientMode.NATIVE:
            self.search_client = AzureSQLNativeVectorSearchClient(connection_string, table_prefix)
        else:
            raise ValueError(f"Query Mode: {query_mode} not supported")

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def from_texts(
            cls: Type[AzureSQLVectorDB],
            texts: List[str],
            embeddings: Embeddings,
            metadata_list: Optional[List[dict]] = None,
            query_mode: VectorSearchClientMode = VectorSearchClientMode.CLASSIC,
            distance_metric: DistanceMetric = DistanceMetric.DOT_PRODUCT,
            table_prefix: str = "sql_embeddings",
            connection_string: Optional[str] = None,
    ) -> AzureSQLVectorDB:

        instance = cls(
            embeddings,
            query_mode=query_mode,
            distance_metric=distance_metric,
            table_prefix=table_prefix,
            connection_string=connection_string,
        )
        instance.add_texts(texts, metadata_list, embeddings.embed_documents(texts))
        return instance

    def add_texts(
            self,
            texts: Iterable[str],
            metadata_list: Optional[List[dict]] = None,
            embeddings: Optional[List[List[float]]] = None,
            **kwargs: Any,
    ) -> List[str]:

        for i, content_text in enumerate(texts):
            content = content_text
            metadata = metadata_list[i]
            embedding = embeddings[i]

            self.search_client.insert_row(content, metadata, embedding)

        return []

    def similarity_search(
            self, query: str, k: int = 4, filters: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filters
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(self, query: str, k: int = 4, filters: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:

        # Creates embedding vector from user query
        vector: list[float] = self.embedding.embed_query(query)

        return self.__similarity_search_with_score(vector, k=k, filters=filters)

    def __similarity_search_with_score(
            self, query_vector: list[float], k: int = 4, filters: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query. Uses cosine similarity.

        Args:
            query_vector: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        operation = self.distance_metric
        result = []
        search_results = self.search_client.compute_similarity(query_vector, k=k,
                                                               similarity_operation=operation, filters=filters)
        for result in search_results:
            page_content = result['content']
            metadata = result['metadata']
            score = float(result['distance_score'])

            doc = Document(page_content=page_content, metadata=metadata)
            result.append((doc, score))

        return result


# AzureSQLVectorDBRetriever is not needed, but we keep it for backwards compatibility
AzureSQLVectorDBRetriever = VectorStoreRetriever
