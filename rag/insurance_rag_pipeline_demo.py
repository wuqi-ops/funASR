"""
保险语音客服场景下的 RAG 教学案例。

这个案例专门演示你接下来要学习的 4 个点：
1. Query 理解
   - 意图分类
   - 实体抽取
   - 问题改写
2. 混合召回
   - 关键词召回
   - 向量召回
   - 结构化召回
3. 重排体系
   - 模型分数和业务规则如何融合
4. 评估治理
   - 离线评估
   - 简单线上监控
   - 人工闭环入口

注意：
- 这是“生产思路教学版”，不是完整生产实现
- 重点是把生产级设计思路拆成可运行、可理解的代码
- 为了便于学习，很多能力都做了简化版实现

安装依赖：
    ./.venv/bin/pip install sentence-transformers chromadb

运行方式：
    ./.venv/bin/python rag/insurance_rag_pipeline_demo.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class QueryAnalysis:
    """
    Query 理解结果。

    你可以把它理解成：
    “原始 ASR 文本经过理解后，抽出的结构化信息”
    """

    original_query: str
    rewritten_query: str
    intent: str
    entities: dict


def build_insurance_documents() -> list[dict]:
    """
    构造一个迷你版保险知识库。

    为了模拟生产级场景，每条文档都带了 metadata。
    这些 metadata 在真实项目里很重要，因为：
    - 检索前要过滤
    - 重排时要加权
    - 生成答案时要引用
    - 评估时要看知识源质量
    """
    return [
        {
            "id": "doc-1",
            "title": "尊享e生2024版-等待期条款",
            "content": (
                "尊享e生2024版重大疾病责任等待期为90天。等待期内首次确诊的重大疾病，"
                "不承担重大疾病保险金给付责任。"
            ),
            "metadata": {
                "domain": "claim",
                "product": "尊享e生2024版",
                "topic": "等待期",
                "region": "全国",
                "channel": "全渠道",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "条款原文",
                "priority": 1.0,
                "access_role": "customer",
            },
        },
        {
            "id": "doc-2",
            "title": "尊享e生2024版-门急诊责任",
            "content": (
                "尊享e生2024版普通门急诊责任在保单责任范围内可赔付，"
                "是否赔付以责任范围、免赔额、医院范围及具体责任约定为准。"
            ),
            "metadata": {
                "domain": "claim",
                "product": "尊享e生2024版",
                "topic": "门急诊",
                "region": "全国",
                "channel": "全渠道",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "条款原文",
                "priority": 1.0,
                "access_role": "customer",
            },
        },
        {
            "id": "doc-3",
            "title": "重疾险核保-智能核保规则",
            "content": (
                "重疾险支持智能核保时，客户可根据健康告知进行在线核保评估。"
                "若不支持智能核保，则需人工核保。"
            ),
            "metadata": {
                "domain": "underwriting",
                "product": "通用重疾险",
                "topic": "智能核保",
                "region": "全国",
                "channel": "线上",
                "version": "通用",
                "effective": "2024-01-01",
                "source_type": "规则手册",
                "priority": 0.9,
                "access_role": "agent",
            },
        },
        {
            "id": "doc-4",
            "title": "理赔FAQ-住院前门急诊是否赔付",
            "content": (
                "住院前门急诊是否赔付，需要结合具体产品责任、就诊时间、医院范围和条款约定判断，"
                "不能仅依据口头描述直接确认。"
            ),
            "metadata": {
                "domain": "claim",
                "product": "通用医疗险",
                "topic": "住院前门急诊",
                "region": "全国",
                "channel": "全渠道",
                "version": "通用",
                "effective": "2024-01-01",
                "source_type": "FAQ",
                "priority": 0.7,
                "access_role": "customer",
            },
        },
        {
            "id": "doc-5",
            "title": "上海地区-特需部责任说明",
            "content": (
                "上海地区部分高端医疗险产品可覆盖特需部责任，具体以产品责任范围和医院清单为准。"
            ),
            "metadata": {
                "domain": "claim",
                "product": "高端医疗险",
                "topic": "特需部",
                "region": "上海",
                "channel": "高端渠道",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "产品说明书",
                "priority": 0.8,
                "access_role": "customer",
            },
        },
        {
            "id": "doc-6",
            "title": "客户服务SOP-高风险问题转人工",
            "content": (
                "涉及理赔结论、拒赔原因、核保结论等高风险问题时，客服机器人不得直接给出确定性承诺，"
                "应提示以人工审核或正式核赔结论为准。"
            ),
            "metadata": {
                "domain": "service",
                "product": "通用",
                "topic": "转人工",
                "region": "全国",
                "channel": "全渠道",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "SOP",
                "priority": 1.0,
                "access_role": "agent",
            },
        },
        {
            "id": "doc-7",
            "title": "监管口径-禁止超条款承诺",
            "content": (
                "客服答复应以正式条款、规则文件和公司制度为依据，"
                "不得做超出条款范围的赔付承诺或收益承诺。"
            ),
            "metadata": {
                "domain": "compliance",
                "product": "通用",
                "topic": "合规",
                "region": "全国",
                "channel": "全渠道",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "监管制度",
                "priority": 1.0,
                "access_role": "agent",
            },
        },
        {
            "id": "doc-8",
            "title": "保全FAQ-线上变更联系方式",
            "content": (
                "客户可通过官方 App 或小程序发起联系方式变更，"
                "需完成身份校验并以系统审核结果为准。"
            ),
            "metadata": {
                "domain": "policy_service",
                "product": "通用",
                "topic": "联系方式变更",
                "region": "全国",
                "channel": "线上",
                "version": "2024",
                "effective": "2024-01-01",
                "source_type": "FAQ",
                "priority": 0.8,
                "access_role": "customer",
            },
        },
    ]


def build_search_text(item: dict) -> str:
    """
    构造适合 embedding 检索的文本。

    生产里通常不会只拿正文做向量化，
    还会把标题、主题、产品名等信息拼进去。
    """
    metadata = item["metadata"]
    return (
        f"标题：{item['title']}。"
        f"产品：{metadata['product']}。"
        f"主题：{metadata['topic']}。"
        f"领域：{metadata['domain']}。"
        f"内容：{item['content']}"
    )


def classify_intent(query: str) -> str:
    """
    用关键词做一个简化版意图分类。

    在生产里，这里通常会是：
    - 规则模型
    - 小分类模型
    - 或 LLM + 规则兜底
    """
    rules = {
        "claim": ["赔", "理赔", "报销", "住院", "门诊", "等待期", "免赔额"],
        "underwriting": ["核保", "健康告知", "带病", "能买吗", "投保"],
        "policy_service": ["保全", "变更", "联系方式", "退保", "续保"],
        "compliance": ["投诉", "合规", "监管", "承诺"],
    }
    for intent, keywords in rules.items():
        if any(keyword in query for keyword in keywords):
            return intent
    return "service"


def extract_entities(query: str) -> dict:
    """
    抽取关键实体。

    这里做了最小可理解版：
    - 产品名
    - 地区
    - 主题词
    - 风险级别
    """
    products = ["尊享e生2024版", "高端医疗险", "重疾险"]
    regions = ["上海", "北京", "全国"]
    topics = ["等待期", "门急诊", "智能核保", "特需部", "联系方式变更"]

    found_product = next((item for item in products if item in query), None)
    found_region = next((item for item in regions if item in query), None)
    found_topic = next((item for item in topics if item in query), None)

    high_risk_keywords = ["能赔吗", "一定赔吗", "能不能赔", "拒赔", "核保结论"]
    risk_level = "high" if any(word in query for word in high_risk_keywords) else "normal"

    return {
        "product": found_product,
        "region": found_region,
        "topic": found_topic,
        "risk_level": risk_level,
    }


def rewrite_query(query: str, intent: str, entities: dict) -> str:
    """
    把口语化 query 改写成更适合检索的 query。

    生产里这一步非常常见，
    因为客户说法往往很口语，不适合直接拿去检索。
    """
    parts = [query]
    if intent == "claim":
        parts.append("重点关注理赔责任、等待期、责任范围、赔付条件")
    elif intent == "underwriting":
        parts.append("重点关注核保规则、健康告知、智能核保")
    elif intent == "policy_service":
        parts.append("重点关注保全流程、线上办理、身份校验")

    if entities.get("product"):
        parts.append(f"产品是{entities['product']}")
    if entities.get("region"):
        parts.append(f"地区是{entities['region']}")
    if entities.get("topic"):
        parts.append(f"主题是{entities['topic']}")

    return "；".join(parts)


def understand_query(query: str) -> QueryAnalysis:
    """
    聚合 Query 理解的 3 个动作：
    - 意图分类
    - 实体抽取
    - 问题改写
    """
    intent = classify_intent(query)
    entities = extract_entities(query)
    rewritten_query = rewrite_query(query, intent, entities)
    return QueryAnalysis(
        original_query=query,
        rewritten_query=rewritten_query,
        intent=intent,
        entities=entities,
    )


def build_vector_store(model: SentenceTransformer, documents: list[dict]):
    """
    构建本地向量库。
    """
    db_path = Path(__file__).resolve().parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))

    collection_name = "insurance_rag_pipeline_demo"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    search_texts = [build_search_text(item) for item in documents]
    embeddings = model.encode(search_texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=[item["id"] for item in documents],
        documents=search_texts,
        embeddings=embeddings,
        metadatas=[item["metadata"] for item in documents],
    )
    return collection


def tokenize(text: str) -> list[str]:
    """
    非严格中文分词，只做教学用途。

    规则：
    - 提取连续中文片段
    - 提取连续英文和数字片段
    """
    return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", text.lower())


def keyword_recall(query_analysis: QueryAnalysis, documents: list[dict], top_k: int) -> list[dict]:
    """
    关键词召回。

    用 token overlap 模拟 BM25 这一类关键词检索的效果。
    真实项目里，这里通常会接 Elasticsearch / OpenSearch。
    """
    query_tokens = set(tokenize(query_analysis.rewritten_query))
    results = []
    for item in documents:
        doc_text = build_search_text(item)
        doc_tokens = set(tokenize(doc_text))
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            results.append(
                {
                    "id": item["id"],
                    "route": "keyword",
                    "keyword_score": overlap / max(len(query_tokens), 1),
                }
            )

    results.sort(key=lambda row: row["keyword_score"], reverse=True)
    return results[:top_k]


def vector_recall(
    query_analysis: QueryAnalysis,
    collection,
    top_k: int,
) -> list[dict]:
    """
    向量召回。
    """
    query_embedding = model.encode(
        query_analysis.rewritten_query,
        normalize_embeddings=True,
    ).tolist()
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    results = []
    for doc_id, distance in zip(raw_results["ids"][0], raw_results["distances"][0]):
        results.append(
            {
                "id": doc_id,
                "route": "vector",
                "vector_score": 1.0 - float(distance),
            }
        )
    return results


def structured_recall(query_analysis: QueryAnalysis, documents: list[dict]) -> list[dict]:
    """
    结构化召回。

    保险场景很依赖结构化过滤：
    - 产品
    - 地区
    - 领域
    - 权限
    - 版本

    这里做一个简化版：
    按产品 / 地区 / 意图进行“硬条件或半硬条件”召回。
    """
    entities = query_analysis.entities
    results = []
    for item in documents:
        score = 0.0
        metadata = item["metadata"]

        if metadata["domain"] == query_analysis.intent:
            score += 0.5
        if entities.get("product") and metadata["product"] == entities["product"]:
            score += 0.3
        if entities.get("region") and metadata["region"] in [entities["region"], "全国"]:
            score += 0.2
        if entities.get("topic") and metadata["topic"] == entities["topic"]:
            score += 0.3

        if score > 0:
            results.append(
                {
                    "id": item["id"],
                    "route": "structured",
                    "structured_score": score,
                }
            )

    results.sort(key=lambda row: row["structured_score"], reverse=True)
    return results[:5]


def merge_recall_results(*recall_groups: list[dict]) -> dict[str, dict]:
    """
    合并多路召回结果。

    同一个文档可能被多路召回同时命中，
    合并后统一进入重排。
    """
    merged = {}
    for group in recall_groups:
        for item in group:
            row = merged.setdefault(
                item["id"],
                {
                    "id": item["id"],
                    "keyword_score": 0.0,
                    "vector_score": 0.0,
                    "structured_score": 0.0,
                    "routes": set(),
                },
            )
            row["routes"].add(item["route"])
            if "keyword_score" in item:
                row["keyword_score"] = max(row["keyword_score"], item["keyword_score"])
            if "vector_score" in item:
                row["vector_score"] = max(row["vector_score"], item["vector_score"])
            if "structured_score" in item:
                row["structured_score"] = max(row["structured_score"], item["structured_score"],)
    return merged


def source_type_bonus(source_type: str) -> float:
    """
    知识源加分。

    保险生产里，知识源可信度常常比“语义相似”更重要。
    """
    bonuses = {
        "条款原文": 0.20,
        "规则手册": 0.15,
        "监管制度": 0.18,
        "SOP": 0.10,
        "产品说明书": 0.08,
        "FAQ": 0.05,
    }
    return bonuses.get(source_type, 0.0)


def risk_control_bonus(query_analysis: QueryAnalysis, doc: dict) -> float:
    """
    高风险问题时，更偏向正式知识源。
    """
    if query_analysis.entities["risk_level"] != "high":
        return 0.0

    if doc["metadata"]["source_type"] in {"条款原文", "规则手册", "监管制度"}:
        return 0.10
    return -0.05


def rerank(
    query_analysis: QueryAnalysis,
    merged_results: dict[str, dict],
    documents_by_id: dict[str, dict],
) -> list[dict]:
    """
    重排阶段。

    这里把“模型分数”和“业务规则”结合起来。

    final_score =
        0.35 * vector_score
      + 0.20 * keyword_score
      + 0.20 * structured_score
      + 0.10 * source_type_bonus
      + 0.10 * priority
      + 0.05 * risk_control_bonus

    这就是保险 RAG 里常见的思路：
    - 模型分数提供语义能力
    - 业务规则保证可控性和可信度
    """
    reranked = []
    for doc_id, item in merged_results.items():
        doc = documents_by_id[doc_id]
        metadata = doc["metadata"]

        source_bonus = source_type_bonus(metadata["source_type"])
        priority = float(metadata["priority"])
        risk_bonus = risk_control_bonus(query_analysis, doc)

        final_score = (
            0.35 * item["vector_score"]
            + 0.20 * item["keyword_score"]
            + 0.20 * item["structured_score"]
            + 0.10 * source_bonus
            + 0.10 * priority
            + 0.05 * risk_bonus
        )

        reranked.append(
            {
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "metadata": metadata,
                "routes": sorted(item["routes"]),
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"],
                "structured_score": item["structured_score"],
                "source_bonus": source_bonus,
                "priority": priority,
                "risk_bonus": risk_bonus,
                "final_score": final_score,
            }
        )

    reranked.sort(key=lambda row: row["final_score"], reverse=True)
    return reranked


def build_answer(query_analysis: QueryAnalysis, reranked_docs: list[dict]) -> dict:
    """
    用重排后的 Top 文档拼一个“客服推荐答复”。

    这里不接 LLM，只做一个可控模板。
    生产里这一步通常是：
    - 先把证据喂给 LLM
    - 再做回答约束和引用控制
    """
    top_docs = reranked_docs[:2]
    citations = [item["title"] for item in top_docs]

    if query_analysis.entities["risk_level"] == "high":
        answer = (
            "当前问题涉及高风险保险结论，建议优先参考正式条款或规则文件，"
            "并提示以人工审核或正式核赔结论为准。"
        )
    elif top_docs:
        answer = (
            f"根据已召回知识，优先建议参考《{top_docs[0]['title']}》。"
            "如需对客户回复，应结合产品版本、责任范围和适用条件说明。"
        )
    else:
        answer = "当前未检索到足够证据，建议转人工进一步处理。"

    return {
        "answer": answer,
        "citations": citations,
    }


def offline_evaluate(
    documents: list[dict],
    collection,
) -> dict:
    """
    做一个最小可运行版离线评估。

    评估关注两个关键指标：
    - recall_at_3: 正确文档是否在 Top3 内
    - mrr: 正确文档排名是否靠前
    """
    eval_set = [
        {
            "query": "尊享e生2024版等待期多久，刚买一个月能赔吗",
            "gold_doc_id": "doc-1",
        },
        {
            "query": "住院前门急诊能不能报销",
            "gold_doc_id": "doc-4",
        },
        {
            "query": "重疾险支持智能核保吗",
            "gold_doc_id": "doc-3",
        },
        {
            "query": "上海特需部能赔吗",
            "gold_doc_id": "doc-5",
        },
    ]

    documents_by_id = {item["id"]: item for item in documents}
    hit_count = 0
    reciprocal_rank_sum = 0.0

    for sample in eval_set:
        analysis = understand_query(sample["query"])
        keyword_hits = keyword_recall(analysis, documents, top_k=5)
        vector_hits = vector_recall(analysis, collection, top_k=5)
        structured_hits = structured_recall(analysis, documents)
        merged = merge_recall_results(keyword_hits, vector_hits, structured_hits)
        reranked = rerank(analysis, merged, documents_by_id)
        top_ids = [item["id"] for item in reranked[:3]]

        if sample["gold_doc_id"] in top_ids:
            hit_count += 1

        for rank, item in enumerate(reranked, start=1):
            if item["id"] == sample["gold_doc_id"]:
                reciprocal_rank_sum += 1.0 / rank
                break

    total = len(eval_set)
    return {
        "recall_at_3": hit_count / total,
        "mrr": reciprocal_rank_sum / total,
        "eval_count": total,
    }


def online_monitor(logs: list[dict]) -> dict:
    """
    简化版线上监控。

    生产里会接监控平台，这里只演示思路：
    - 高风险问题占比
    - 低证据命中次数
    - 转人工次数
    """
    total = len(logs)
    high_risk_count = sum(1 for item in logs if item["risk_level"] == "high")
    low_confidence_count = sum(1 for item in logs if item["top_score"] < 0.45)
    human_handoff_count = sum(1 for item in logs if item["need_human"] is True)

    return {
        "total_requests": total,
        "high_risk_ratio": high_risk_count / total if total else 0.0,
        "low_confidence_ratio": low_confidence_count / total if total else 0.0,
        "human_handoff_ratio": human_handoff_count / total if total else 0.0,
    }


def human_feedback_loop(logs: list[dict]) -> list[dict]:
    """
    模拟人工闭环队列。

    这些 case 在生产里通常会进入：
    - 标注平台
    - 质检平台
    - 知识运营平台
    """
    review_queue = []
    for item in logs:
        if item["need_human"] or item["top_score"] < 0.45:
            review_queue.append(
                {
                    "query": item["query"],
                    "reason": "高风险或低置信度，需要人工复核",
                    "predicted_doc": item["top_doc_title"],
                }
            )
    return review_queue


def print_query_analysis(analysis: QueryAnalysis) -> None:
    print("一、Query 理解")
    print(f"原始 Query: {analysis.original_query}")
    print(f"意图分类: {analysis.intent}")
    print(f"实体抽取: {analysis.entities}")
    print(f"问题改写: {analysis.rewritten_query}")
    print()


def print_recall(
    keyword_hits: list[dict],
    vector_hits: list[dict],
    structured_hits: list[dict],
) -> None:
    print("二、混合召回")
    print(f"关键词召回: {keyword_hits}")
    print(f"向量召回: {vector_hits}")
    print(f"结构化召回: {structured_hits}")
    print()


def print_rerank(reranked: list[dict]) -> None:
    print("三、重排结果")
    for rank, item in enumerate(reranked[:3], start=1):
        print(
            f"{rank}. {item['title']} | final={item['final_score']:.4f} | "
            f"routes={item['routes']} | vector={item['vector_score']:.4f} | "
            f"keyword={item['keyword_score']:.4f} | structured={item['structured_score']:.4f}"
        )
    print()


def main() -> None:
    documents = build_insurance_documents()
    documents_by_id = {item["id"]: item for item in documents}

    model_name = "BAAI/bge-m3"
    print(f"正在加载模型: {model_name}")
    global model
    model = SentenceTransformer(model_name)
    print("模型加载完成。\n")

    collection = build_vector_store(model, documents)

    queries = [
        "尊享e生2024版刚买一个月，现在查出来重疾能赔吗",
        "上海高端医疗险的特需部能不能报",
        "重疾险支持智能核保吗",
        "我想在线变更一下联系方式",
    ]

    online_logs = []
    route_counter = Counter()

    for query in queries:
        print("=" * 100)
        analysis = understand_query(query)
        print_query_analysis(analysis)

        # 关键词召回（BM25）
        keyword_hits = keyword_recall(analysis, documents, top_k=5)
        # 向量召回
        vector_hits = vector_recall(analysis, collection, top_k=5)
        # 结构化召回
        structured_hits = structured_recall(analysis, documents)
        print_recall(keyword_hits, vector_hits, structured_hits)

        # 合并多路召回结果
        merged = merge_recall_results(keyword_hits, vector_hits, structured_hits)
        # 重排阶段（把“模型分数”和“业务规则”结合起来）
        reranked = rerank(analysis, merged, documents_by_id)
        print_rerank(reranked)

        answer = build_answer(analysis, reranked)
        print("四、推荐答复")
        print(answer["answer"])
        print(f"引用依据: {answer['citations']}")
        print()

        top_item = reranked[0] if reranked else None
        need_human = (
            analysis.entities["risk_level"] == "high"
            or top_item is None
            or top_item["final_score"] < 0.45
        )
        if top_item:
            for route in top_item["routes"]:
                route_counter[route] += 1

        online_logs.append(
            {
                "query": query,
                "risk_level": analysis.entities["risk_level"],
                "top_score": top_item["final_score"] if top_item else 0.0,
                "need_human": need_human,
                "top_doc_title": top_item["title"] if top_item else "无",
            }
        )

    print("=" * 100)
    print("五、离线评估")
    offline_metrics = offline_evaluate(documents, collection)
    print(offline_metrics)
    print()

    print("六、线上监控")
    monitor_metrics = online_monitor(online_logs)
    monitor_metrics["top1_route_distribution"] = dict(route_counter)
    print(monitor_metrics)
    print()

    print("七、人工闭环")
    review_queue = human_feedback_loop(online_logs)
    for item in review_queue:
        print(item)
    print()

    print("学习总结：")
    print("1. Query 理解负责把口语问题变成可检索、可路由的结构化问题。")
    print("2. 混合召回负责同时兼顾术语精确命中、语义相似和业务过滤。")
    print("3. 重排把模型分数和保险业务规则融合，保证结果更稳。")
    print("4. 评估治理让系统可监控、可复盘、可持续优化。")


if __name__ == "__main__":
    main()
