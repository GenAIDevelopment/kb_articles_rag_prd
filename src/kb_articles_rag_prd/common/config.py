from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

@dataclass
class KnowledgeBaseIndexerConfig:
    """Knowledge Base Indexer Config
    """
    source_type: Literal["local", "url"] = field(default="local")

    source: str = field(default="data/")

    use_multithreading: bool = field(default=False)

    glob: Optional[str] = field(default="**/*.txt")

    vector_store_engine: Literal["chroma", "faiss"] = field(default="chroma")

    vector_store_persist_path: str = field(default="vectordb/")

    vector_store_collection_name: Optional[str] = field(default="kbarticles")

    record_manager_db_url: str = field(default="sqlite:///record_manager.db")

    record_manager_namespace: Optional[str] = field(default="kbarticles")

    embedding_engine: Literal["openai", "huggingface", "vertex"] = field(default="vertex")

    embedding_engine_model: str = field(default="text-embedding-005")

    source_id_key: str = field(default="source")

    clean_up: str = field(default="incremental")

    chunk_size: int = field(default=1000)

    chunk_overlap: int = field(default=100)

    seperators: Tuple[str, ...] = field(default=("\n\n", "\n", " ", ""))