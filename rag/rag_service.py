"""
总结服务类： 用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from sentence_transformers import CrossEncoder


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        # 初始化重排序模型（第一次运行会自动下载模型到 ~/.cache/huggingface/）
        self.reranker = CrossEncoder('D:/Agent项目实战/model/bge-reranker-v2-m3')
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self , query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def retrieve_docs_with_rerank(self, query: str, top_k_initial: int = 10, top_k_final: int = 3) -> list[Document]:
        candidates = self.retriever.invoke(query)
        if len(candidates) <= top_k_final:
            return candidates
        candidates_to_rerank = candidates[:top_k_initial]

        # 【新增】打印重排序前的 Top-3
        print("===== 重排序前 Top-3 =====")
        for i, doc in enumerate(candidates_to_rerank[:3]):
            print(f"{i + 1}. {doc.page_content[:100]}...")
            print(f"   来源: {doc.metadata.get('source', 'unknown')}\n")

        pairs = [[query, doc.page_content] for doc in candidates_to_rerank]
        scores = self.reranker.predict(pairs)  # 原是 compute_score
        scored_docs = sorted(zip(candidates_to_rerank, scores), key=lambda x: x[1], reverse=True)

        # 【新增】打印重排序后的 Top-3
        print("===== 重排序后 Top-3 =====")
        for i, (doc, score) in enumerate(scored_docs[:3]):
            print(f"{i + 1}. (相关性分数: {score:.4f}) {doc.page_content[:100]}...")
            print(f"   来源: {doc.metadata.get('source', 'unknown')}\n")

        return [doc for doc, _ in scored_docs[:top_k_final]]

    def rag_summarize(self, query: str) -> str:
        # 使用带重排序的方法获取文档
        context_docs = self.retrieve_docs_with_rerank(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("登山的基本条件"))