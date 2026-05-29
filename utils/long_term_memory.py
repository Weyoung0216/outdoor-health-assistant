import os
from langchain_chroma import Chroma
from model.factory import embed_model
from utils.path_tool import get_abs_path

LTM_ROOT = get_abs_path("long_term_memory")

def get_user_memory_store(user_id: str) -> Chroma:
    """获取或创建对应用户的长期记忆向量库"""
    persist_dir = os.path.join(LTM_ROOT, user_id)
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        collection_name=f"user_{user_id}_memories",
        embedding_function=embed_model,
        persist_directory=persist_dir,
    )

def add_memory(user_id: str, text: str, metadata: dict = None):
    """将一条记忆文本存入长期记忆"""
    if metadata is None:
        metadata = {}
    metadata["user_id"] = user_id
    store = get_user_memory_store(user_id)
    store.add_texts([text], metadatas=[metadata])

def retrieve_memories(user_id: str, query: str, k: int = 3) -> list[str]:
    """检索与 query 相关的长期记忆文本列表"""
    store = get_user_memory_store(user_id)
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]