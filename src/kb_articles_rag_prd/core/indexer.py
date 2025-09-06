"""
This module has necessary classes defined to perform indexing
"""
from abc import abstractmethod, ABC
from langchain.indexes import index, IndexingResult, SQLRecordManager
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from kb_articles_rag_prd.common.config import KnowledgeBaseIndexerConfig




class BaseIndexer(ABC):
    """This is base implementation of indexer

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, config: KnowledgeBaseIndexerConfig):
        """Initialize indexer

        Args:
            config (KnowledgeBaseIndexerConfig): _description_
        """
        self._config = config
        self._setup()

    def _setup(self):
        """This method sets up necessary objects
        for indexing
        """
        # setup record manager
        self._setup_record_manager()
        self._setup_vector_store()
        self._setup_docs()

    def _setup_docs(self):
        """This method will setup the docs
        """
        if self._config.source_type == "local":
            directory_loader = DirectoryLoader(
                path=self._config.source,
                glob=self._config.glob,
                use_multithreading=self._config.use_multithreading
            )
            self._docs = directory_loader.load()

        else:
            raise NotImplementedError(f"source {self._config.source} is Not implemented yet")

    def _setup_vector_store(self):
        """This method setup the vector store
        """
        if self._config.embedding_engine == "vertex":
            embedding = VertexAIEmbeddings(
                model_name=self._config.embedding_engine_model
            )
        else:
            raise NotImplementedError(f"Embedding model {self._config.embedding_engine} is not implemented yet")
        self._vector_store = Chroma(
            collection_name=self._config.vector_store_collection_name,
            embedding_function=embedding,
            persist_directory=self._config.vector_store_persist_path
        )


    def _setup_record_manager(self):
        """This method setsup the record manager
        """
        namespace = self._config.record_manager_namespace
        db_url = self._config.record_manager_db_url
        self._record_manager: SQLRecordManager = SQLRecordManager(
            namespace=namespace,
            db_url=db_url
        )
        self._record_manager.create_schema()

    @abstractmethod
    def index(self, *args, **kwargs) -> IndexingResult:
        """This method indexes the data into vector database
        and gives the indexing result
        """

