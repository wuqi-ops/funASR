"""
一个适合 ASR 文本知识库学习的 Embedding Demo。

学习目标：
1. 理解什么是“把文本向量化”
2. 理解如何比较 query 和文档的语义相似度
3. 理解 Embedding 在 ASR 后处理/术语召回中的一个最小可运行例子

结论先说：
- 可以使用 bge-m3 进行向量化
- bge-m3 对中英文、多语言、长短文本都比较友好
- 它既适合做语义检索，也适合做 RAG / 知识库召回的第一阶段

安装依赖：
    pip install sentence-transformers torch

首次运行会自动下载模型：
    BAAI/bge-m3
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    手写一个余弦相似度函数，帮助你理解“两个向量为什么能比较相似度”。

    公式：
        cos(a, b) = (a · b) / (|a| * |b|)

    如果前面已经把向量做了 L2 归一化，那么点积就近似等于余弦相似度。
    这里仍然保留完整公式，便于学习。
    """
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def print_vector_shape(name: str, vector: np.ndarray) -> None:
    """
    打印向量维度，帮助建立“文本 -> 向量数组”的直观认识。
    """
    print(f"{name} 向量维度: {vector.shape}")


def main() -> None:
    # 1) 选择模型
    # bge-m3 是一个常见的多语言 embedding 模型。
    # 对你当前的学习目标来说，它完全可以用来做“文本知识库向量化”。
    #
    # 为什么适合当前 demo：
    # - 支持中文
    # - 支持语义检索
    # - 社区使用广泛，资料较多
    # - 适合先学习 dense embedding（稠密向量）
    model_name = "BAAI/bge-m3"

    print(f"正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    # 2) 准备一个“迷你 ASR 文本知识库”
    # 这些文本可以理解为：术语解释、纠错候选、领域知识片段。
    documents = [
        "Embedding 是把文本映射为高维向量，用于语义检索和相似度计算。",
        "向量数据库用于存储向量，并支持高效的近邻检索。",
        "召回阶段负责找出候选文档，重排阶段负责把最相关结果排在前面。",
        "ASR 后处理可以结合知识库，对专业术语和专有名词进行纠错。",
        "bge-m3 是一个多语言文本向量模型，适合检索、RAG 和知识库场景。",
        "如果语音识别把向量数据库识别错了，可以结合知识库做术语纠偏。",
    ]

    # 3) 模拟一个 ASR 输出文本
    # 假设识别结果不够标准，或者包含口语化表达。
    query = "我想学习一下向量数据库和召回重排是怎么配合工作的"

    print(f"Query: {query}\n")
    print("知识库文档:")
    for idx, doc in enumerate(documents, start=1):
        print(f"{idx}. {doc}")
    print()

    # 4) 向量化
    # normalize_embeddings=True 的作用：
    # - 把向量归一化到长度约等于 1
    # - 后续用点积/余弦相似度时更方便
    #
    # 这里你可以把 encode 理解成：
    # “把人类可读的文本，转换成机器可计算的语义坐标”
    query_embedding = model.encode(query, normalize_embeddings=True)
    document_embeddings = model.encode(documents, normalize_embeddings=True)

    print_vector_shape("Query", query_embedding)
    print(f"文档向量矩阵维度: {document_embeddings.shape}\n")

    # 5) 计算 query 和每个文档的相似度
    # 在真实检索系统中，这一步通常由向量库代劳。
    # 当前 demo 为了便于学习，直接在内存中计算。
    scored_results: list[tuple[int, str, float]] = []
    for idx, doc_embedding in enumerate(document_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scored_results.append((idx, documents[idx], score))

    # 6) 按相似度从高到低排序
    # 这一步可以理解为“最简单版的召回结果排序”。
    scored_results.sort(key=lambda item: item[2], reverse=True)

    print("Top 3 召回结果:")
    for rank, (_, doc, score) in enumerate(scored_results[:3], start=1):
        print(f"{rank}. score={score:.4f}")
        print(f"   {doc}")
    print()

    # 7) 再模拟一个“ASR 纠错型” query
    # 这里的重点是：用户说得不标准，或者 ASR 识别得不完全准确，
    # 但 embedding 仍然有机会把它映射到正确语义附近。
    correction_query = "语音识别把象量数据库识别错了，怎么纠正"
    correction_embedding = model.encode(correction_query, normalize_embeddings=True)

    correction_results: list[tuple[int, str, float]] = []
    for idx, doc_embedding in enumerate(document_embeddings):
        score = cosine_similarity(correction_embedding, doc_embedding)
        correction_results.append((idx, documents[idx], score))

    correction_results.sort(key=lambda item: item[2], reverse=True)

    print(f"ASR 纠错型 Query: {correction_query}\n")
    print("Top 3 召回结果:")
    for rank, (_, doc, score) in enumerate(correction_results[:3], start=1):
        print(f"{rank}. score={score:.4f}")
        print(f"   {doc}")
    print()

    # 8) 学习总结
    print("学习总结：")
    print("1. Embedding 的作用，是把文本转成可以比较的向量。")
    print("2. 相似度越高，说明两个文本在语义空间里越接近。")
    print("3. 在 ASR 场景里，可以用它做术语召回、错词纠偏、知识增强。")
    print("4. 当前 demo 只展示了 dense embedding，已经足够你入门。")
    print("5. 后续你可以继续接：向量库存储 -> TopK 召回 -> 重排模型。")


if __name__ == "__main__":
    main()
