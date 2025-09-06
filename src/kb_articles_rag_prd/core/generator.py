from abc import ABC, abstractmethod
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from kb_articles_rag_prd.common.config import KnowledgeBaseGenConfig

class BaseGenerator(ABC):
    """Base Generator implementation
    """

    def __init__(self, config: KnowledgeBaseGenConfig):
        self._config = config
        self._setup()

    def _setup(self):
        """Setup method
        """
        self._setup_retriever()
        self._setup_model()

    def _setup_model(self):
        """Setup model
        """
        self._model = init_chat_model(
            model=self._config.model,
            model_provider=self._config.model_provier
        )

    def _setup_retriever(self):
        """setup retriever
        """
        # get embedding engine
        if self._config.embedding_engine == "openai":
            # from langchain_openai import OpenAIEmbeddings
            # self._embedding_engine = OpenAIEmbeddings(
            #     model=self._config.embedding_engine_model
            # )
            raise NotImplementedError("OpenAIEmbeddings is not implemented yet")
        elif self._config.embedding_engine == "huggingface":
            # from langchain_community.embeddings import HuggingFaceEmbeddings
            # self._embedding_engine = HuggingFaceEmbeddings(
            #     model_name=self._config.embedding_engine_model
            # )
            raise NotImplementedError("HuggingFaceEmbeddings is not implemented yet")
        elif self._config.embedding_engine == "vertex":
            from langchain_google_vertexai import VertexAIEmbeddings
            self._embedding_engine = VertexAIEmbeddings(
                model_name=self._config.embedding_engine_model
            )
        if self._config.vector_store_engine == "chroma":
            from langchain_chroma import Chroma
            self._vector_store = Chroma(
                persist_directory=self._config.vector_store_persist_path,
                embedding_function=self._embedding_engine,
                collection_name=self._config.vector_store_collection_name
            )

    @abstractmethod
    def respond(self, question:str) -> BaseMessage:
        pass


