import json
import math
import re
from collections import Counter
from typing import Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from langgraph.graph import START, StateGraph

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Hybrid retrieval: each retriever proposes CANDIDATE_K chunks; Reciprocal
# Rank Fusion picks the FINAL_K best. The wide candidate pool lets a chunk
# that one retriever ranks highly survive a mediocre rank from the other
# (e.g. "Gold status" ranks 6.3 Tier Levels #1 in BM25 but only #11 in
# vector search). FINAL_K is 6 rather than 4 so that one retriever's top
# pick still makes the cut when the other retriever only contributes noise
# (e.g. "pet"/"cat" never appear in the corpus, so BM25 is noise there and
# would otherwise crowd out vector's #1, 7.5 Service Animals).
CANDIDATE_K = 12
FINAL_K = 6
RRF_K = 60


class KnowledgeBaseSection(TypedDict):
    id: str  # structured section number, e.g. "6.3"
    title: str  # e.g. "Tier Levels"
    text: str


class KnowledgeBaseEntry(TypedDict):
    url: str
    title: str
    sections: List[KnowledgeBaseSection]


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


DECOMPOSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Break this customer service question into independent sub-queries that each target a single topic in an airline reference manual (e.g. baggage policy, loyalty benefits, rebooking rules).

If the question only covers one topic, return it as-is.

Return ONLY a JSON array of strings, nothing else.

Question: {question}
""",
)


SYSTEM_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are the customer service assistant for SkyWay Airlines. Answer the passenger's question using ONLY the policy excerpts retrieved from the SkyWay Customer Service Reference Manual below.

Guidelines:
- Be polite, warm, and helpful. Address the passenger directly.
- When addressing a customer whose name is not known, them as 'Passenger' or 'Customer' as is appropriate in the situation.
- Be precise: state fees, deadlines, weights, distances, and compensation amounts exactly as they appear in the excerpts.
- This is a single exchange — give a complete answer; do not ask follow-up questions unless you need additional clarification from the customer.
- If the excerpts do not contain the information needed, say so honestly and refer the passenger to the SkyWay Customer Service Centre (+49 69 555 0100, 24/7). Never invent policies, fees, or exceptions that are not in the excerpts.
- Keep the answer concise and easy to scan (short paragraphs or bullets).

Question: {question}

RETRIEVED POLICY EXCERPTS:
{context}

Answer:
""",
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Minimal in-memory BM25 keyword index over the corpus documents.

    Complements the embedding-based retriever: exact terms like "Gold" or
    "PIR" score highly here even when they are diluted in the embedding."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._doc_tokens = [Counter(_tokenize(doc.page_content)) for doc in documents]
        self._doc_lengths = [sum(tf.values()) for tf in self._doc_tokens]
        self._avg_doc_length = sum(self._doc_lengths) / len(documents)
        self._doc_frequencies: Counter = Counter()
        for tf in self._doc_tokens:
            self._doc_frequencies.update(tf.keys())

    def _score(self, query_terms: List[str], doc_index: int) -> float:
        tf = self._doc_tokens[doc_index]
        length_norm = (
            1 - self.b + self.b * self._doc_lengths[doc_index] / self._avg_doc_length
        )
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            df = self._doc_frequencies[term]
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            score += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * length_norm)
        return score

    def top(self, query: str, k: int) -> List[Document]:
        query_terms = _tokenize(query)
        scored = sorted(
            range(len(self.documents)),
            key=lambda i: self._score(query_terms, i),
            reverse=True,
        )
        return [self.documents[i] for i in scored[:k]]


def _rrf_fuse(ranked_lists: List[List[Document]], k: int) -> List[Document]:
    """Reciprocal Rank Fusion: merge ranked lists into one, rewarding
    documents that rank well in any list."""
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata["section_id"]  # unique per subsection chunk
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    best = sorted(scores, key=lambda s: scores[s], reverse=True)[:k]
    return [by_key[key] for key in best]


def load_knowledge_base(json_path: str) -> List[KnowledgeBaseEntry]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(entries: List[KnowledgeBaseEntry]) -> List[Document]:
    """One Document per subsection, so each numbered subsection of the
    manual (e.g. "6.3. Tier Levels") is its own retrieval chunk. The
    structured section id travels in the metadata so retrieval results
    can be checked against required sources without parsing headings."""
    documents: List[Document] = []
    for entry in entries:
        for section in entry["sections"]:
            documents.append(
                Document(
                    page_content=f"{section['id']}. {section['title']}\n\n{section['text']}",
                    metadata={
                        "url": entry["url"],
                        "section_id": section["id"],
                        "section_title": section["title"],
                    },
                )
            )
    return documents


def build_rag_graph(
    knowledge_base_path: str,
    llm_model: str = "openai/gpt-oss-120b:free",
    embedding_model: str = "openai/text-embedding-3-small",
    openrouter_api_key: str = "",
):
    entries = load_knowledge_base(knowledge_base_path)
    documents = create_documents(entries)

    api_key = SecretStr(openrouter_api_key)
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )
    llm = ChatOpenAI(
        model=llm_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=documents)
    bm25_index = BM25Index(documents)

    def decompose_query(question: str) -> List[str]:
        response = llm.invoke(DECOMPOSE_PROMPT.invoke({"question": question}))
        try:
            sub_queries = json.loads(str(response.content))
            if isinstance(sub_queries, list) and all(
                isinstance(q, str) for q in sub_queries
            ):
                return sub_queries
        except (json.JSONDecodeError, TypeError):
            pass
        return [question]

    def retrieve(state: State):
        sub_queries = decompose_query(state["question"])
        print(f"  Sub-queries: {sub_queries}")
        all_ranked_lists: List[List[Document]] = []
        for query in sub_queries:
            all_ranked_lists.append(
                vector_store.similarity_search(query, k=CANDIDATE_K)
            )
            all_ranked_lists.append(bm25_index.top(query, k=CANDIDATE_K))
        fused = _rrf_fuse(all_ranked_lists, k=FINAL_K)
        return {"context": fused}

    def generate(state: State):
        if not state["context"]:
            return {"answer": "I don't know."}
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = SYSTEM_PROMPT.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def answer_question(graph, question: str) -> Dict[str, object]:
    result = graph.invoke({"question": question})
    retrieved_docs = result.get("context", [])
    return {
        "response": result["answer"],
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "retrieved_sections": [
            {
                "id": doc.metadata["section_id"],
                "title": doc.metadata["section_title"],
            }
            for doc in retrieved_docs
        ],
    }
