"""
一个更贴近当前项目的 ASR 术语纠错 Demo。

它模拟的是这样一条链路：
1. 先准备一份 ASR 术语知识库
2. 把每个术语转成适合检索的文本
3. 用向量库先召回 TopK 候选
4. 再结合别名命中、领域权重做重排
5. 输出“最可能的正确术语”

这个 Demo 更接近真实 ASR 后处理场景：
- 用户说的是语音
- ASR 输出里可能有同音词、近音词、口语表达
- 我们通过术语库去“拉回正确术语”

安装依赖：
    ./.venv/bin/pip install sentence-transformers chromadb

运行方式：
    ./.venv/bin/python rag/asr_term_rerank_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def load_term_kb() -> list[dict]:
    """
    从 JSON 文件读取术语知识库。

    真实项目里，这份数据可以来自：
    - 产品术语表
    - 业务词库
    - 热词词典
    - 历史纠错样本
    """
    kb_path = Path(__file__).resolve().parent / "asr_terms.json"
    with kb_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_search_text(item: dict) -> str:
    """
    给每个术语构造一段“适合检索的文本”。

    为什么不能只存 term？
    因为真实查询里，用户可能说的是：
    - 别名
    - 口语化说法
    - 错别字
    - 更长的上下文描述

    所以我们把：
    - 标准术语
    - 别名
    - 描述
    拼成一条检索文本，便于召回阶段命中。
    """
    aliases_text = " ".join(item["aliases"])
    return f"{item['term']} {aliases_text} {item['description']}"


def build_vector_store(model: SentenceTransformer):
    """
    构建本地向量库，并写入术语知识库。
    """
    terms = load_term_kb()
    search_texts = [build_search_text(item) for item in terms]

    db_path = Path(__file__).resolve().parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))

    collection_name = "asr_term_rerank_demo"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [item["id"] for item in terms]
    metadatas = [
        {
            "term": item["term"],
            "topic": item["topic"],
            "weight": item["weight"],
        }
        for item in terms
    ]

    embeddings = model.encode(search_texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        documents=search_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection, terms


def alias_hit_score(query: str, item: dict) -> float:
    """
    计算“别名命中分数”。

    这一步非常贴近 ASR 纠错：
    如果 query 里出现了某个术语的别名、近似写法或常见错写，
    就给它额外加分。
    """
    lowered_query = query.lower()

    if item["term"].lower() in lowered_query:
        return 1.0

    for alias in item["aliases"]:
        if alias.lower() in lowered_query:
            return 1.0

    return 0.0


def char_overlap_score(query: str, item: dict) -> float:
    """
    计算 query 和术语文本的字符重合度。

    这里仍然保持简单，
    目的是让你先理解“重排可以综合多种特征”。
    """
    query_chars = {char for char in query if char.strip()}
    candidate_text = item["term"] + "".join(item["aliases"])
    candidate_chars = {char for char in candidate_text if char.strip()}

    if not query_chars:
        return 0.0

    return len(query_chars & candidate_chars) / len(query_chars)


def topic_bonus(query: str, item: dict) -> float:
    """
    按 query 的主题给候选术语加一点业务分。

    例如：
    - 提到“向量库”，更偏向 vector_db
    - 提到“流式”“实时”，更偏向 streaming
    - 提到“模型”“向量化”，更偏向 embedding_model / rag
    """
    rules = [
        (["向量库", "数据库", "chroma"], "vector_db"),
        (["重排", "召回", "排序"], "retrieval"),
        (["流式", "实时", "websocket", "chunk"], "streaming"),
        (["向量化", "embedding", "嵌入"], "rag"),
        (["模型", "bge", "m3"], "embedding_model"),
    ]

    lowered_query = query.lower()
    for keywords, target_topic in rules:
        if any(keyword.lower() in lowered_query for keyword in keywords):
            if item["topic"] == target_topic:
                return 0.15
    return 0.0


def rerank_candidates(query: str, recall_results: dict, terms_by_id: dict) -> list[dict]:
    """
    对召回候选做第二阶段重排。

    打分由 4 部分组成：
    - semantic_score: 向量召回的语义分
    - alias_score: 是否命中标准术语或别名
    - lexical_score: 字符重合度
    - business_score: 主题业务加分
    """
    reranked = []

    for doc_id, doc, metadata, distance in zip(
        recall_results["ids"][0],
        recall_results["documents"][0],
        recall_results["metadatas"][0],
        recall_results["distances"][0],
    ):
        item = terms_by_id[doc_id]

        semantic_score = 1.0 - float(distance)
        alias_score = alias_hit_score(query, item)
        lexical_score = char_overlap_score(query, item)
        business_score = topic_bonus(query, item)
        term_weight = float(metadata.get("weight", 1.0))

        final_score = (
            0.50 * semantic_score
            + 0.20 * alias_score
            + 0.20 * lexical_score
            + 0.10 * business_score
        ) * term_weight

        reranked.append(
            {
                "id": doc_id,
                "term": item["term"],
                "aliases": item["aliases"],
                "topic": item["topic"],
                "doc": doc,
                "semantic_score": semantic_score,
                "alias_score": alias_score,
                "lexical_score": lexical_score,
                "business_score": business_score,
                "term_weight": term_weight,
                "final_score": final_score,
            }
        )

    reranked.sort(key=lambda row: row["final_score"], reverse=True)
    return reranked


def print_recall_results(results: dict, terms_by_id: dict) -> None:
    """
    打印第一阶段召回结果。
    """
    print("第一阶段：术语召回结果")
    for rank, (doc_id, distance) in enumerate(
        zip(results["ids"][0], results["distances"][0]),
        start=1,
    ):
        item = terms_by_id[doc_id]
        print(
            f"{rank}. term={item['term']}, "
            f"distance={distance:.4f}, "
            f"aliases={item['aliases']}"
        )
    print()


def print_rerank_results(results: list[dict]) -> None:
    """
    打印第二阶段重排结果。
    """
    print("第二阶段：术语重排结果")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. term={item['term']}, "
            f"final={item['final_score']:.4f}, "
            f"semantic={item['semantic_score']:.4f}, "
            f"alias={item['alias_score']:.4f}, "
            f"lexical={item['lexical_score']:.4f}, "
            f"business={item['business_score']:.4f}"
        )
        print(f"   aliases={item['aliases']}")
    print()


def run_one_case(query: str, collection, model: SentenceTransformer, terms_by_id: dict) -> None:
    """
    运行一条 ASR 纠错案例。

    你可以把 query 理解成：
    - ASR 初始转写结果
    - 或者用户口语化描述
    """
    print("=" * 80)
    print(f"ASR 文本: {query}\n")

    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    recall_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    print_recall_results(recall_results, terms_by_id)

    reranked_results = rerank_candidates(query, recall_results, terms_by_id)
    print_rerank_results(reranked_results[:3])

    best = reranked_results[0]
    print(f"最终推荐术语: {best['term']}")
    print()


def main() -> None:
    model_name = "BAAI/bge-m3"
    print(f"正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    collection, terms = build_vector_store(model)
    terms_by_id = {item["id"]: item for item in terms}

    # 下面这些 query 都故意写得更像“真实 ASR 输出”。
    # 有些是口语化表达，有些是近似写法，有些包含上下文。
    queries = [
        "我想用 bgm3 做文本向量化",
        "我准备用 chroma 搭一个向量库",
        "流式识别里 websocket 和 chunk 怎么配合",
        "语音识别之后想做重排和召回",
        "我现在在看 fun asr 和 sense voice small 模型",
    ]

    for query in queries:
        run_one_case(query, collection, model, terms_by_id)

    print("学习总结：")
    print("1. 这类术语库更贴近 ASR 后处理和领域词纠偏。")
    print("2. 召回阶段先找出可能的正确术语。")
    print("3. 重排阶段再结合别名命中和业务规则精排。")
    print("4. 如果后面接到真实 ASR 结果，就可以直接复用这条链路。")
    print("5. 下一步可以继续扩展成批量评估和术语纠错服务。")


if __name__ == "__main__":
    main()
