"""
RAG pipeline for financial-document Q&A.

Key responsibilities
-------------------
1. Parse user questions into structured metadata filters (Pydantic)
2. Retrieve relevant chunks from Pinecone (dense + sparse RRF)
3. Build a prompt with conversation history and retrieved context
4. Stream an answer from Azure OpenAI
"""

import os
import logging
from collections import deque
from operator import itemgetter
from typing import Any, Dict, Iterable, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationInfo, field_validator, validator

# LangChain / OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Pinecone + SPLADE
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone_text.sparse import SpladeEncoder

# Caching
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
load_dotenv()

AZURE_OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER   = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_MODEL        = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOYMENT_NAME   = os.getenv("CHAT_DEPLOYMENT_NAME")
CHAT_MODEL             = os.getenv("CHAT_MODEL_NAME", "gpt-4o")

PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
DENSE_INDEX_NAME       = os.getenv("PINECONE_DENSE_INDEX_NAME")
SPARSE_INDEX_NAME      = os.getenv("PINECONE_SPARSE_INDEX_NAME")

TOTAL_RETRIEVED_DOC_CHUNKS = int(os.getenv("TOTAL_RETRIEVED_DOC_CHUNKS", "8"))
CHUNKS_PER_DOC             = int(os.getenv("CHUNKS_PER_DOC", "5"))
TOP_K_PER_STAGE            = max(6, TOTAL_RETRIEVED_DOC_CHUNKS * 2)

MAX_MEM_TOKENS = 80_000          # LLM-memory budget (tokens)
USER_QUERIES: deque[str] = deque(maxlen=5)      # last 5 user prompts
LAST_FILTER: Dict[str, Dict[str, Any]] = {}     # sticky metadata filter

set_llm_cache(InMemoryCache())

# ---------------------------------------------------------------------------
# Constants used for validation and prompt-hints
# ---------------------------------------------------------------------------
ALLOWED_TYPES  = {"sec_filing", "earnings_call_transcript", "company_profile"}
ALLOWED_FORMS  = {"10-K", "10-Q"}
YEAR_RANGE     = (1900, 2100)
ALLOWED_QTRS   = {1, 2, 3, 4}

TYPES_TXT   = ", ".join(sorted(ALLOWED_TYPES))
FORMS_TXT   = ", ".join(sorted(ALLOWED_FORMS))
YEAR_TXT    = f"{YEAR_RANGE[0]}–{YEAR_RANGE[1]}"
QUARTER_TXT = ", ".join(map(str, sorted(ALLOWED_QTRS)))

FORM_MAP = {"10K": "10-K", "10-K": "10-K", "10Q": "10-Q", "10-Q": "10-Q"}


def _norm_form(v: str | None) -> str | None:
    """Canonicalize form strings (e.g. 10k → 10-K)."""
    if not v:
        return None
    return FORM_MAP.get(v.upper().replace(" ", "").replace("-", ""), v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _merge_filters(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Union-merge two Pinecone filter dictionaries.
    Scalars are promoted to {"$in": [...]} when needed.
    """
    merged = {**old}
    for k, v in new.items():
        if _is_effectively_empty({k: v}):
            continue
        old_v = merged.get(k)

        if isinstance(old_v, dict) and "$in" in old_v and isinstance(v, dict) and "$in" in v:
            merged[k] = {"$in": list(set(old_v["$in"]) | set(v["$in"]))}
        elif isinstance(v, dict) and "$in" in v:
            merged[k] = v if old_v is None else {"$in": list({old_v, *v["$in"]})}
        elif isinstance(old_v, dict) and "$in" in old_v:
            merged[k] = {"$in": list(set(old_v["$in"]) | {v})}
        elif old_v is not None and old_v != v:
            merged[k] = {"$in": [old_v, v]}
        else:
            merged[k] = v
    return merged


def build_fused_query() -> str:
    """Fuse the last ≤5 user prompts (newline-separated)."""
    return "\n".join(USER_QUERIES)


def _is_effectively_empty(f: Dict[str, Any]) -> bool:
    """Return True if a filter imposes no real constraints."""
    if not f:
        return True
    sym = f.get("symbol")
    sym_ok = sym in (None, "", [], {"$in": []})
    others_present = any(k for k in f if k != "symbol")
    return sym_ok and not others_present


# ---------------------------------------------------------------------------
# Pydantic model for structured filter extraction
# ---------------------------------------------------------------------------
class QueryFilter(BaseModel):
    # string fields
    symbol:  List[str] | str | None = None
    type:    List[str] | str | None = None
    form:    List[str] | str | None = None
    # numeric fields
    year:    List[int] | int | None = None
    quarter: List[int] | int | None = None

    # --- string normalizers ------------------------------------------------
    @validator("symbol", pre=True, always=True)
    def _to_list_upper(cls, v):
        if v is None:
            return []
        return [s.upper() for s in (v if isinstance(v, list) else [v])]

    @validator("type", pre=True)
    def _norm_type(cls, v):
        if v is None:
            return []
        items = [v] if isinstance(v, str) else v
        bad = [x for x in items if x not in ALLOWED_TYPES]
        if bad:
            raise ValueError(f"Unknown type(s): {', '.join(bad)}")
        return items

    @validator("form", pre=True)
    def _norm_form_field(cls, v):
        if v is None:
            return []
        items = [_norm_form(v)] if isinstance(v, str) else [_norm_form(x) for x in v]
        bad = [x for x in items if x not in ALLOWED_FORMS]
        if bad:
            raise ValueError(f"Unknown form(s): {', '.join(bad)}")
        return items

    # --- numeric normalizers ----------------------------------------------
    @field_validator("year", "quarter", mode="before")
    def _to_int_list(cls, v, info: ValidationInfo):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        ints = [int(x) for x in items]
        if info.field_name == "quarter" and any(q not in ALLOWED_QTRS for q in ints):
            raise ValueError("quarter must be between 1 and 4")
        return ints

    # --- conversion to Pinecone filter ------------------------------------
    def to_pinecone_filter(self) -> Dict[str, Any]:
        def one_or_in(seq: List[Any] | None):
            if not seq:
                return None
            return seq[0] if len(seq) == 1 else {"$in": seq}

        return {
            k: v
            for k, v in {
                "symbol": one_or_in(self.symbol),
                "type": one_or_in(self.type),
                "form": one_or_in(self.form),
                "year": one_or_in(self.year),
                "quarter": one_or_in(self.quarter),
            }.items()
            if v is not None
        }


# ---------------------------------------------------------------------------
# Extraction chain (LLM → JSON → Pydantic)
# ---------------------------------------------------------------------------
_parser = PydanticOutputParser(pydantic_object=QueryFilter)

_extract_prompt = PromptTemplate(
    template=r"""
Task  
Extract the metadata filters explicitly requested in the question.

Return valid **JSON only** matching this schema:
{format}

Allowed values  
• "type"    → {{{types}}}  
• "form"    → {{{forms}}}  
• "year"    → integers {years}  
• "quarter" → {quarters}  
• "symbol"  → stock tickers

Always use arrays when multiple values are present.

User question:
"{question}"
""",
    input_variables=["question"],
    partial_variables={
        "format": _parser.get_format_instructions(),
        "types": TYPES_TXT,
        "forms": FORMS_TXT,
        "years": YEAR_TXT,
        "quarters": QUARTER_TXT,
    },
)

_extract_llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=CHAT_DEPLOYMENT_NAME,
    model_name=CHAT_MODEL,
    api_version=AZURE_OPENAI_API_VER,
    temperature=0,
)

extract_chain = _extract_prompt | _extract_llm | _parser

# ---------------------------------------------------------------------------
# Retrieval utilities
# ---------------------------------------------------------------------------
def rrf_merge(dense: List[Document], sparse: List[Document], k: int = 20) -> List[Document]:
    """Reciprocal-rank fusion of two result lists."""
    scores: Dict[str, float] = {}
    seen: Dict[str, Document] = {}

    for rank, doc in enumerate(dense):
        key = doc.metadata.get("id") or id(doc)
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        seen[key] = doc

    for rank, doc in enumerate(sparse):
        key = doc.metadata.get("id") or id(doc)
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        seen[key] = doc

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [seen[key] for key, _ in ranked]


def make_retrieve_fn(
    dense_store: PineconeVectorStore,
    sparse_index,
    sparse_encoder,
) -> RunnableLambda:
    """
    Factory returning a Runnable that:
      1. Extracts metadata filters
      2. Searches dense + sparse indexes with progressive relaxation
      3. Fuses results with RRF
    """

    # ---- sparse helper ---------------------------------------------------
    def _sparse(q_text: str, flt: Dict[str, Any]) -> List[Document]:
        sv = sparse_encoder.encode_queries([q_text])[0]
        res = sparse_index.query(
            sparse_vector=sv,
            top_k=TOP_K_PER_STAGE,
            include_metadata=True,
            filter=flt or None,
        )
        return [Document(page_content=m["metadata"]["text"], metadata=m["metadata"]) for m in res["matches"]]

    # ---- dense+sparse search with fallback levels ------------------------
    def pinecone_search(q_text: str, base_filter: Dict[str, Any]) -> tuple[list[Document], list[Document]]:
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=base_filter or None)
        if dense:
            return dense, _sparse(q_text, base_filter)

        f2 = {k: v for k, v in base_filter.items() if k != "quarter"}
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=f2 or None)
        if dense:
            return dense, _sparse(q_text, f2)

        f3 = {k: v for k, v in f2.items() if k != "year"}
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=f3 or None)
        if dense:
            return dense, _sparse(q_text, f3)

        return dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE), _sparse(q_text, {})

    # ---- main retrieval callable -----------------------------------------
    def _retrieve(fused_query: str, convo_id: str = "default") -> List[Document]:
        qf = extract_chain.invoke({"question": fused_query})
        fresh_flt = qf.to_pinecone_filter()

        base_flt = _merge_filters(LAST_FILTER.get(convo_id, {}), fresh_flt)
        dense_docs, sparse_docs = pinecone_search(fused_query, base_flt)
        merged = rrf_merge(dense_docs, sparse_docs, k=TOP_K_PER_STAGE * 2)

        # per-filing caps
        by_blob: Dict[str, List[Document]] = {}
        for d in merged:
            key = d.metadata.get("symbol") + "|" + d.metadata.get("blob_name")
            by_blob.setdefault(key, []).append(d)

        per_filing = [d for docs in by_blob.values() for d in docs[:CHUNKS_PER_DOC]]
        final = per_filing[:TOTAL_RETRIEVED_DOC_CHUNKS]

        if merged:
            LAST_FILTER[convo_id] = base_flt
        return final

    return RunnableLambda(lambda q: _retrieve(q))


# ---------------------------------------------------------------------------
# Memory for chat history
# ---------------------------------------------------------------------------
memory_llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=CHAT_DEPLOYMENT_NAME,
    model_name=CHAT_MODEL,
    api_version=AZURE_OPENAI_API_VER,
)

memory = ConversationTokenBufferMemory(
    llm=memory_llm,
    max_token_limit=MAX_MEM_TOKENS,
    human_prefix="User",
    ai_prefix="Assistant",
    memory_key="history",
    return_messages=True,
)

# ---------------------------------------------------------------------------
# Build RAG chain (single global instance)
# ---------------------------------------------------------------------------
def build_rag_chain():
    embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VER,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    dense_index = pc.Index(DENSE_INDEX_NAME)
    sparse_index = pc.Index(SPARSE_INDEX_NAME)

    dense_store = PineconeVectorStore(index=dense_index, embedding=embedding, namespace="")
    sparse_encoder = SpladeEncoder()
    retrieve_fn = make_retrieve_fn(dense_store, sparse_index, sparse_encoder)

    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=CHAT_DEPLOYMENT_NAME,
        model_name=CHAT_MODEL,
        api_version=AZURE_OPENAI_API_VER,
        max_tokens=1024,
        temperature=0.25,
    )

    prompt = PromptTemplate.from_template(
        """
        You are an intelligent assistant specializing in financial and investment analysis.

        Conversation so far:
        {history}

        Guidelines:
        - Provide clear, informative answers.
        - Use as much detail as needed.
        - Cite documents in-text, e.g. (AAPL 10-K 2024).
        - Ignore unrelated ads or boilerplate.
        - Remain professional and concise.

        Question: {question}

        Context:
        {context}

        Answer:
        """
    )

    return (
        {
            "context": RunnableLambda(lambda _: build_fused_query()) | retrieve_fn
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": itemgetter("prompt_question"),
            "history": RunnableLambda(lambda _: memory.load_memory_variables({})["history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


RAG_CHAIN = build_rag_chain()

# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------
def stream_answer(question: str) -> Iterable[str]:
    """
    Stream the assistant's answer while updating
    user-query history and memory buffer.
    """
    USER_QUERIES.append(question)

    inputs = {"prompt_question": question}
    chunks: List[str] = []

    for piece in RAG_CHAIN.stream(inputs):
        text = piece if isinstance(piece, str) else piece.content
        chunks.append(text)
        yield text

    memory.save_context({"input": question}, {"output": "".join(chunks)})




