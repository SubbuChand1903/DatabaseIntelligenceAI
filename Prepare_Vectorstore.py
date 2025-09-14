import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SimpleField,
    LexicalAnalyzerName,
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_search_index():
    """
    Deletes (if exists) and recreates the Azure AI Search index with the right schema + vector config.
    """

    # --- Env ---
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)

    endpoint = "https://azureai-search-0011.search.windows.net"
    key = "evCKlgtIaQMCcmUZ3oNlQlQXrz98rB2RphxR5m74UqAzSeDBs8tx"
    index_name = "dw-impact-analysis-index-new"

    if not endpoint or not key:
        raise ValueError("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in environment or .env")

    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # --- Delete existing (if present) ---
    try:
        index_client.get_index(index_name)
        logging.info(f"Index '{index_name}' exists. Deleting...")
        index_client.delete_index(index_name)
        logging.info(f"Index '{index_name}' deleted.")
    except ResourceNotFoundError:
        logging.info(f"Index '{index_name}' not found. No deletion needed.")

    # --- Fields ---
    fields = [
        # Key
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, retrievable=True),

        # Main text
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
            analyzer_name=LexicalAnalyzerName.STANDARD_LUCENE,
        ),

        # Metadata
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        SearchableField(name="source_file", type=SearchFieldDataType.String, filterable=True, searchable=True, retrievable=True),
        SearchableField(name="object_name", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="table_name", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SimpleField(name="schema_name", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        # Make columns filterable so you can do $filter=columns eq 'IsAG'
        SearchableField(name="columns", type=SearchFieldDataType.String, searchable=True, filterable=True, retrievable=True),
        SimpleField(name="change_type", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        SimpleField(name="object_type", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
        SimpleField(name="metadata", type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True, retrievable=True),

        # Vector field (must be searchable=True for vector search)
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,  # REQUIRED for vector fields
            # (Do not set 'retrievable' here; this model class doesn't accept it in this SDK version)
            vector_search_dimensions=384,  # all-MiniLM-L6-v2
            vector_search_profile_name="v1",
        ),
    ]

    # --- Vector search config (HNSW) ---
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw",
                parameters=HnswParameters(
                    m=20,                # typical: 16–48
                    ef_construction=400,
                    ef_search=200,       # 100–200 is a good start
                    metric="cosine",
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="v1",
                algorithm_configuration_name="hnsw",
            )
        ],
    )

    # --- Create index ---
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        scoring_profiles=[],
        cors_options=None,
    )

    logging.info(f"Creating new index '{index_name}' with vector dimension 384...")
    result = index_client.create_index(index)
    logging.info(f"Index '{result.name}' created with {len(result.fields)} fields.")

    # Log fields for quick verification
    for field in result.fields:
        logging.info(f"  - {field.name} ({field.type})")

    logging.info("Index setup completed successfully.")


if __name__ == "__main__":
    setup_search_index()
