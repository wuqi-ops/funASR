from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


def build_example_text() -> list[dict[str, str]]:
    return [
        {
            "id": "doc-1",
            "text": "Embedding 是把文本映射成向量，便于做语义相似度计算。",
            "metadata": {"topic": "embedding", "source": "kb"},
        },
        {
            "id": "doc-2",
            "text": "向量数据库可以存储文本向量，并支持近邻检索。",
            "metadata": {"topic": "vector_db", "source": "kb"},
        },
        {
            "id": "doc-3",
            "text": "召回阶段先找候选文档，重排阶段再把最相关结果排在前面。",
            "metadata": {"topic": "retrieval", "source": "kb"},
        },
        {
            "id": "doc-4",
            "text": "ASR 后处理可以结合术语库，对专有名词和专业词进行纠偏。",
            "metadata": {"topic": "asr", "source": "kb"},
        },
        {
            "id": "doc-5",
            "text": "bge-m3 是一个多语言 embedding 模型，适合知识库检索场景。",
            "metadata": {"topic": "embedding", "source": "kb"},
        },
        {
            "id": "doc-6",
            "text": "如果语音识别把向量数据库识别错了，可以用知识库召回正确术语。",
            "metadata": {"topic": "asr", "source": "kb"},
        },
    ]


def main() -> None:
    # 加载模型
    model_name = "BAAI/bge-m3"
    print(f"正在加载 embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    # 材料向量化
    documents = build_example_text()
    ids = [item["id"] for item in documents]
    texts = [item["text"] for item in documents]
    metadata = [item["metadata"] for item in documents]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # 创建vector db
    collection_name = "asr_self_demo"
    db_path = Path(__file__).resolve().parent / "chroma_db"
    vector_db_client = chromadb.PersistentClient(db_path)
    try:
        vector_db_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = vector_db_client.create_collection(collection_name)

    # 写入向量库
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata,)

    # 构建问题向量
    query = "我想学习向量数据库在召回里的作用"
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # 检索
    results = collection.query(query_embeddings=[query_embedding], where_document={"$contains": "数据库"}, n_results=2)
    print(results)


if __name__ == "__main__":
    main()
