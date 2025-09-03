from langchain.indexes import SQLRecordManager, index
from langchain_core.indexing.api import IndexingResult
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from kb_articles_rag_prd.core.readers import DirectoryReader



def test_indexing() -> IndexingResult:
    namespace = "testing"
    sql_record_manager = SQLRecordManager(
        namespace=namespace,
        db_url="sqlite:///records.db"
    )
    sql_record_manager.create_schema()
    embeddings = VertexAIEmbeddings(model="text-embedding-005")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./vectordb",
    )
    directory_reader = DirectoryReader()
    docs = directory_reader.read(
        path=r"C:\khajaclassroom\GenerativeAI\agenticai\aug25\kb_articles_rag_prd\src\kb_articles_rag_prd\data",
        glob="*.txt")
    index_stats = index(
        docs,
        record_manager=sql_record_manager,
        vector_store=vector_store,
        cleanup="incremental"  # can be "full" or "incremental"
    )
    return index_stats
    

