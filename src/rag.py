import json
from typing import Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class HandbookEntry(TypedDict):
    url: str
    title: str
    sections: Dict[str, str]


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


SYSTEM_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Act as a conversational interface for answering questions based on the content of the handbook in your knowledge base.

Question: {question}

BELOW IS THE RETRIEVED CONTEXT DATA:
Context: {context}

Answer:
""",
)


def load_handbook(json_path: str) -> List[HandbookEntry]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(entries: List[HandbookEntry]) -> List[Document]:
    documents: List[Document] = []
    for entry in entries:
        metadata = {"url": entry["url"], "title": entry["title"]}
        article_text = ""
        for heading, text in entry["sections"].items():
            article_text += f"\n\n{heading}\n\n{text}"
        documents.append(Document(page_content=article_text, metadata=metadata))
    return documents


def build_rag_graph(
    handbook_path: str,
    llm_model: str = "openai/gpt-oss-120b:free",
    embedding_model: str = "openai/text-embedding-3-small",
    openrouter_api_key: str = "",
):
    entries = load_handbook(handbook_path)
    documents = create_documents(entries)

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=openrouter_api_key,
    )
    llm = ChatOpenAI(
        model=llm_model,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=openrouter_api_key,
    )

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=documents)

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

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
    }
