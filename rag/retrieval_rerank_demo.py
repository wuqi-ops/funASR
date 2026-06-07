"""
一个适合初学者的“召回 + 重排”最小案例。

学习目标：
1. 理解为什么检索系统通常不是“一步到位”，而是分成两阶段
2. 理解召回和重排各自负责什么
3. 理解在 ASR 场景里，为什么“语义相似”还不够，还要叠加业务规则

本示例使用：
- Embedding 模型：BAAI/bge-m3
- 向量库：Chroma
- 重排方式：规则重排（便于学习原理，先不引入额外模型）

安装依赖：
    ./.venv/bin/pip install sentence-transformers chromadb

运行方式：
    ./.venv/bin/python rag/retrieval_rerank_demo.py
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def build_demo_documents() -> list[dict]:
    """
    构造一个迷你版知识库。

    为了方便你理解“重排”的价值，这里故意放入：
    - 很像但比较泛的文档
    - 更贴近 ASR 业务的文档
    - 和 query 相关但不完全相同的文档
    """
    return [
        {
            "id": "doc-1",
            "text": "向量数据库用于存储文本向量，并支持近邻检索。",
            "metadata": {"topic": "vector_db", "source": "kb"},
        },
        {
            "id": "doc-2",
            "text": "召回阶段先从大规模知识库里找候选，重排阶段再提高最终结果的准确率。",
            "metadata": {"topic": "retrieval", "source": "kb"},
        },
        {
            "id": "doc-3",
            "text": "ASR 后处理可以结合术语库，对专业词和专有名词进行纠错。",
            "metadata": {"topic": "asr", "source": "kb"},
        },
        {
            "id": "doc-4",
            "text": "如果语音识别把向量数据库识别错了，可以结合知识库召回正确术语，再做纠偏。",
            "metadata": {"topic": "asr", "source": "kb"},
        },
        {
            "id": "doc-5",
            "text": "Embedding 负责把文本映射成向量，是召回阶段的基础能力。",
            "metadata": {"topic": "embedding", "source": "kb"},
        },
        {
            "id": "doc-6",
            "text": "重排不仅可以看语义分数，还可以叠加关键词命中率、业务规则和字段权重。",
            "metadata": {"topic": "rerank", "source": "kb"},
        },
    ]


def build_vector_store(model: SentenceTransformer):
    """
    建立一个本地 Chroma collection，并把示例文档写进去。

    这里显式指定：
    - hnsw:space = cosine
    这样向量距离会更接近“余弦距离”的语义。
    """
    db_path = Path(__file__).resolve().parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))

    collection_name = "retrieval_rerank_demo"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    documents = build_demo_documents()
    ids = [item["id"] for item in documents]
    texts = [item["text"] for item in documents]
    metadatas = [item["metadata"] for item in documents]

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return collection


def char_overlap_score(query: str, doc: str) -> float:
    """
    一个非常简单的“字符重合度”分数。

    为什么不用复杂分词？
    - 你现在的学习重点是理解“重排可以叠加别的特征”
    - 所以先用最容易看懂的方式演示

    返回值范围：
    - 0 到 1
    - 越大表示 query 和 doc 共享的关键字符越多
    """
    query_chars = {ch for ch in query if ch.strip()}
    doc_chars = {ch for ch in doc if ch.strip()}

    if not query_chars:
        return 0.0

    return len(query_chars & doc_chars) / len(query_chars)


def topic_bonus(query: str, metadata: dict) -> float:
    """
    一个非常简单的“业务规则加分”。

    真实项目中，重排不只是看语义分数。
    还可能看：
    - 文档类型
    - 时间新鲜度
    - 权限
    - 用户画像
    - 领域标签

    这里为了学习方便，只做一个小规则：
    - 如果 query 明显在问 ASR 纠错相关问题
    - 并且文档 topic 也是 asr
    - 则给一点业务加分
    """
    asr_keywords = ["语音识别", "ASR", "术语", "纠错", "识别错"]
    is_asr_query = any(keyword in query for keyword in asr_keywords)
    if is_asr_query and metadata.get("topic") == "asr":
        return 0.15
    return 0.0


def rerank_candidates(query: str, recall_results: dict) -> list[dict]:
    """
    对召回候选做二次排序。

    这里使用一个教学型打分公式：
    final_score =
        0.70 * semantic_score
      + 0.20 * lexical_score
      + 0.10 * business_score

    解释：
    - semantic_score: 第一阶段向量召回给出的语义相似度
    - lexical_score: query 和文档的字符重合度
    - business_score: 简单的 ASR 场景加分

    这就是“重排”的核心思想：
    第一阶段先大范围找回来；
    第二阶段再融合更多特征，让排序更贴近业务目标。
    """
    reranked = []

    result_ids = recall_results["ids"][0]
    result_docs = recall_results["documents"][0]
    result_metadatas = recall_results["metadatas"][0]
    result_distances = recall_results["distances"][0]

    for doc_id, doc, metadata, distance in zip(
        result_ids,
        result_docs,
        result_metadatas,
        result_distances,
    ):
        # 在 cosine 距离设定下，distance 越小表示越相似。
        # 为了更符合直觉，这里转成“越大越好”的 semantic_score。
        semantic_score = 1.0 - float(distance)
        lexical_score = char_overlap_score(query, doc)
        business_score = topic_bonus(query, metadata)

        final_score = (
            0.70 * semantic_score
            + 0.20 * lexical_score
            + 0.10 * business_score
        )

        reranked.append(
            {
                "id": doc_id,
                "doc": doc,
                "metadata": metadata,
                "distance": float(distance),
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "business_score": business_score,
                "final_score": final_score,
            }
        )

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked


def print_recall_results(results: dict) -> None:
    """
    打印第一阶段“召回结果”。
    """
    print("第一阶段：召回结果")
    for rank, (doc_id, doc, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"{rank}. id={doc_id}, distance={distance:.4f}, metadata={metadata}")
        print(f"   {doc}")
    print()


def print_rerank_results(results: list[dict]) -> None:
    """
    打印第二阶段“重排结果”。
    """
    print("第二阶段：重排结果")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. id={item['id']}, "
            f"final={item['final_score']:.4f}, "
            f"semantic={item['semantic_score']:.4f}, "
            f"lexical={item['lexical_score']:.4f}, "
            f"business={item['business_score']:.4f}"
        )
        print(f"   {item['doc']}")
    print()


def main() -> None:
    # 1) 加载 embedding 模型
    model_name = "BAAI/bge-m3"
    print(f"正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    # 2) 建立向量库并写入文档
    collection = build_vector_store(model)

    # 3) 准备一个更贴近 ASR 的查询
    query = "语音识别把向量数据库识别错了，怎么通过术语库召回并重新排序"
    print(f"Query: {query}\n")

    # 4) 第一阶段：召回
    # 召回阶段的目标是“别漏掉”。
    # 所以常见做法是先多拿一些候选，比如 Top5 / Top10 / Top20。
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    recall_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    print_recall_results(recall_results)

    # 5) 第二阶段：重排
    # 重排阶段的目标是“让最相关的结果排到最前面”。
    reranked_results = rerank_candidates(query, recall_results)
    print_rerank_results(reranked_results)

    print("学习总结：")
    print("1. 召回负责广撒网，先把可能相关的候选找回来。")
    print("2. 重排负责精排序，让最有用的结果排在最前面。")
    print("3. 重排不仅能看语义分数，还能融合关键词和业务规则。")
    print("4. 在 ASR 场景里，重排尤其适合做术语纠错和领域增强。")
    print("5. 这就是 RAG 中常见的两阶段检索思想。")


if __name__ == "__main__":
    main()
