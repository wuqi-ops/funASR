"""
一个适合初学者的“向量库”最小案例。

学习目标：
1. 理解向量库到底在做什么
2. 理解“写入向量 -> 建库 -> 检索 TopK”的基本流程
3. 理解向量库和 embedding 的分工

你可以把这个 demo 理解为：
- embedding 负责把文本变成向量
- 向量库负责保存这些向量，并在查询时快速找出最相似的内容

本示例使用：
- embedding 模型：BAAI/bge-m3
- 本地向量库：Chroma

安装依赖：
    pip install sentence-transformers torch chromadb

首次运行会自动下载 embedding 模型：
    BAAI/bge-m3
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


def build_demo_documents() -> list[dict]:
    """
    构造一个迷你版的 ASR 文本知识库。

    这里每条数据包含：
    - id: 唯一标识
    - text: 文本内容
    - metadata: 附加信息

    在真实项目里，metadata 很重要。
    它常用于：
    - 按领域过滤，如医疗/客服/教育
    - 按语言过滤，如中文/英文
    - 按数据来源过滤，如 FAQ / 术语表 / 说明书
    """
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
    # 1) 准备 embedding 模型
    # 向量库本身不负责“理解文本语义”，
    # 它只负责存和找。
    # 所以必须先有一个 embedding 模型把文本转成向量。
    model_name = "BAAI/bge-m3"
    print(f"正在加载 embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    # 2) 初始化本地向量库
    # PersistentClient 表示数据会落盘保存，不是程序退出就消失的纯内存库。
    # 这很适合你学习“向量库”时观察数据的持久化行为。
    db_path = Path(__file__).resolve().parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))

    # collection 可以理解成“一个向量表”或“一个知识库集合”。
    # 为了便于重复运行 demo，这里每次都先删除旧 collection，再重建。
    collection_name = "asr_kb_demo"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # 第一次运行时 collection 可能不存在，这里忽略即可。
        pass

    collection = client.create_collection(name=collection_name)

    # 3) 准备要写入向量库的文本数据
    documents = build_demo_documents()
    ids = [item["id"] for item in documents]
    texts = [item["text"] for item in documents]
    metadatas = [item["metadata"] for item in documents]

    print("准备写入如下文本到向量库：")
    for item in documents:
        print(f"- {item['id']}: {item['text']}")
    print()

    # 4) 用 bge-m3 生成文档向量
    # normalize_embeddings=True 让向量更适合后续相似度检索。
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # 5) 写入向量库
    # 这一步是“向量库案例”的核心：
    # - ids: 每条数据的唯一标识
    # - documents: 原始文本
    # - metadatas: 附加字段
    # - embeddings: 对应的向量
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("文本和向量已写入 Chroma。\n")

    # 6) 构造一个查询文本
    # 你可以把它理解为：
    # - 用户的问题
    # - 或者 ASR 输出的一句话
    query = "我想学习向量数据库在召回里的作用"
    print(f"Query: {query}\n")

    # 7) 把 query 也向量化
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # 8) 从向量库中检索最相似的 TopK
    # n_results=3 表示取最相关的 3 条。
    #
    # 这一步就是“向量库的价值”：
    # 如果知识库很大，向量库会比你手工遍历数组更高效。
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    # 9) 打印检索结果
    # Chroma 返回的是一个批量结构，所以这里取第 0 个 query 的结果。
    print("Top 3 检索结果：")
    result_ids = results["ids"][0]
    result_docs = results["documents"][0]
    result_metadatas = results["metadatas"][0]
    result_distances = results["distances"][0]

    for rank, (doc_id, doc, metadata, distance) in enumerate(
        zip(result_ids, result_docs, result_metadatas, result_distances),
        start=1,
    ):
        # Chroma 默认返回的是 distance，值越小一般表示越相近。
        print(f"{rank}. id={doc_id}, distance={distance:.4f}, metadata={metadata}")
        print(f"   {doc}")
    print()

    # 10) 再演示一个更贴近 ASR 的查询
    asr_query = "语音识别把向量数据库说错了，怎么从知识库里找正确术语"
    print(f"ASR Query: {asr_query}\n")

    asr_query_embedding = model.encode(
        asr_query,
        normalize_embeddings=True,
    ).tolist()

    asr_results = collection.query(
        query_embeddings=[asr_query_embedding],
        n_results=3,
    )

    print("Top 3 检索结果：")
    for rank, (doc_id, doc, metadata, distance) in enumerate(
        zip(
            asr_results["ids"][0],
            asr_results["documents"][0],
            asr_results["metadatas"][0],
            asr_results["distances"][0],
        ),
        start=1,
    ):
        print(f"{rank}. id={doc_id}, distance={distance:.4f}, metadata={metadata}")
        print(f"   {doc}")
    print()

    print("学习总结：")
    print("1. embedding 负责把文本变成向量。")
    print("2. 向量库负责保存这些向量，并根据 query 找相似内容。")
    print("3. query 进入系统后，也必须先向量化，才能去检索。")
    print("4. 这就是 RAG / 知识库 / 术语召回系统的最基础流程。")
    print("5. 下一步你可以继续学习 metadata 过滤、混合检索和重排。")


if __name__ == "__main__":
    main()
