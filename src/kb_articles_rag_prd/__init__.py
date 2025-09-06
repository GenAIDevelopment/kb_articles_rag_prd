from kb_articles_rag_prd.common.config import KnowledgeBaseIndexerConfig
from kb_articles_rag_prd.provider.indexer import KBArticleTextIndexer


def create_indexer_config() -> KnowledgeBaseIndexerConfig:
    """This returns sample indexer config

    Returns:
        KnowledgeBaseIndexerConfig: config
    """
    config = KnowledgeBaseIndexerConfig(
        source="data/",
        source_type="local",
        glob="**/*.txt",
        vector_store_engine="chroma",
        vector_store_persist_path="vectordb"
    )
    return config

def indexing_test():
    """Test for indexing implementation
    """
    indexer = KBArticleTextIndexer(
        config=create_indexer_config()
    )
    result = indexer.index()
    print(f"indexing is completed. stats are {result}")

def main():
    """entrypoint
    """
    indexing_test()


if __name__ == "__main__":
    main()