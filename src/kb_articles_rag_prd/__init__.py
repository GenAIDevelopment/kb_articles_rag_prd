from kb_articles_rag_prd.common.config import KnowledgeBaseIndexerConfig, KnowledgeBaseGenConfig
from kb_articles_rag_prd.provider.indexer import KBArticleTextIndexer
from kb_articles_rag_prd.provider.generator import KBArticlesTextGenerator


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

def create_generator_config() -> KnowledgeBaseGenConfig:
    """Return Knowledge base gen config

    Returns:
        KnowledgeBaseGenConfig: knowledge base gen config
    """
    config = KnowledgeBaseGenConfig(
        model="gemini-2.5-flash"
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

def generation_test():
    """Testing for generation"""
    config = create_generator_config()
    text_generator = KBArticlesTextGenerator(config=config)
    question = input('Enter any question ? ')
    response = text_generator.respond(question=question)
    response.pretty_print()



def main():
    """entrypoint
    """
    #indexing_test()
    generation_test()


if __name__ == "__main__":
    main()