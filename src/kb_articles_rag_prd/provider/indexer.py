from langchain.indexes import index, IndexingResult
from kb_articles_rag_prd.core.indexer import BaseIndexer

class KBArticleTextIndexer(BaseIndexer):
    """This indexer expects the docs to be in text format

    Args:
        BaseIndexer (BaseIndexer): _description_
    """

    def index(self, *args, **kwargs) -> IndexingResult:
        """_summary_
        """
        index_stats = index(
            self._docs,
            record_manager=self._record_manager,
            vector_store=self._vector_store,
            cleanup=self._config.clean_up,
            source_id_key=self._config.source_id_key
        )
        return index_stats
        