
from langchain_core.documents import Document
from kb_articles_rag_prd.core.readers import DirectoryReader
from kb_articles_rag_prd.core.test_recordmanager import test_indexing
def read_docs():
    directory_reader = DirectoryReader()
    docs = directory_reader.read(
        path=r"C:\khajaclassroom\GenerativeAI\agenticai\aug25\kb_articles_rag_prd\src\kb_articles_rag_prd\data",
        glob="*.txt")
    for doc in docs:
        print(doc.metadata)


def main() -> None:
    result = test_indexing()
    print(result)
    
