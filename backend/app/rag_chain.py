import os, re, json, functools
from dotenv import load_dotenv
from operator import itemgetter
from typing import Iterable, List, Any, Dict, Literal, Optional
import logging
from collections import deque
import tiktoken

# Pydantic
from pydantic import BaseModel, Field, validator, field_validator, ValidationInfo

# LangChain core
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema import Document
from langchain.memory import ConversationTokenBufferMemory

# LangChain prompts
from langchain.prompts import PromptTemplate

# Azure OpenAI embeddings & chat
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Pinecone + SPLADE
from pinecone import Pinecone              
from langchain_pinecone import PineconeVectorStore    
from pinecone_text.sparse import SpladeEncoder

# LLM cache
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


# Load environment variables once at module import
load_dotenv()

# Config
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
CHAT_DEPLOYMENT_NAME  = os.getenv("CHAT_DEPLOYMENT_NAME")
CHAT_MODEL            = os.getenv("CHAT_MODEL_NAME", "gpt-4o")  # default to gpt-4o

PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
DENSE_INDEX_NAME      = os.getenv("PINECONE_DENSE_INDEX_NAME")
SPARSE_INDEX_NAME     = os.getenv("PINECONE_SPARSE_INDEX_NAME")

# ── retrieval hyper-params (env-configurable) ────────────────────────────
TOTAL_RETRIEVED_DOC_CHUNKS = int(os.getenv("TOTAL_RETRIEVED_DOC_CHUNKS", "8"))
CHUNKS_PER_DOC             = int(os.getenv("CHUNKS_PER_DOC", "5"))

# how many raw candidates to ask Pinecone for (dense & sparse)
# rule of thumb: at least 2× the final budget
TOP_K_PER_STAGE            = max(6, TOTAL_RETRIEVED_DOC_CHUNKS * 2)

logging.info("Retriever config: TOP_K_PER_STAGE=%d  PER_DOC=%d  FINAL=%d",
             TOP_K_PER_STAGE, CHUNKS_PER_DOC, TOTAL_RETRIEVED_DOC_CHUNKS)

# Chat History Config
# Max tokens to be kept in memory for the conversation
MAX_MEM_TOKENS = 80_000 
# Last 5 user queries for retrieval (query fusion)
USER_QUERIES: deque[str] = deque(maxlen=5)         
# Sticky Filter Cache for Retriever
LAST_FILTER: Dict[str, Dict[str, Any]] = {}
# Memory Cache for LLM
set_llm_cache(InMemoryCache())

# (Optional) LangSmith tracing
if os.getenv("LANGSMITH_TRACING_ACTIVE", "false").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Pydantic Schema to do Metadata filter based off user query
FORM_MAP = {
"10K":  "10-K",  "10-K":  "10-K",
"10Q":  "10-Q",  "10-Q":  "10-Q",
}

def _norm_form(v: str | None) -> str | None:
    if not v:
        return None
    v = v.upper().replace(" ", "").replace("-", "")
    return FORM_MAP.get(v, v)

# ─── helper: merge two Pinecone-filter dicts  ───────────────────────────────
def _merge_filters(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Union-merge list-based filters; for scalars fall back to replacement.
    """
    merged = {**old}
    for k, v in new.items():
        if _is_effectively_empty({k: v}):
            continue

        old_v = merged.get(k)
        # ---- both are {"$in": [...] } ----------------------------------
        if isinstance(old_v, dict) and "$in" in old_v \
           and isinstance(v, dict) and "$in" in v:
            merged[k] = {"$in": list(set(old_v["$in"]) | set(v["$in"]))}
        # ---- new is {"$in": [...] }, old scalar ------------------------
        elif isinstance(v, dict) and "$in" in v:
            merged[k] = v if old_v is None else {"$in": list({old_v, *v["$in"]})}
        # ---- new scalar, old {"$in": [...] } ---------------------------
        elif isinstance(old_v, dict) and "$in" in old_v:
            merged[k] = {"$in": list(set(old_v["$in"]) | {v})}
        # ---- both scalars – keep **both** ------------------------------
        elif old_v is not None and old_v != v:
            merged[k] = {"$in": [old_v, v]}
        else:
            merged[k] = v
    return merged



def build_fused_query() -> str:
    """Join up to five most-recent user questions (newline-separated)."""
    return "\n".join(USER_QUERIES)

def _to_list(v) -> list:
    """None → []; scalar → [scalar]; list/{$in:[…]} unchanged."""
    if v is None:
        return []
    if isinstance(v, dict) and "$in" in v:
        v = v["$in"]
    return v if isinstance(v, list) else [v]

def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

ALLOWED_TYPES  = {"sec_filing", "earnings_call_transcript", "company_profile"}
ALLOWED_FORMS  = {"10-K", "10-Q"}
YEAR_RANGE     = (1900, 2100)          # inclusive
ALLOWED_QTRS   = {1, 2, 3, 4}

class QueryFilter(BaseModel):
    # ── strings ───────────────────────────────────────────────
    symbol : List[str] | str | None = None
    type   : List[str] | str | None = None      # multiple allowed
    form   : List[str] | str | None = None
    # ── numbers ───────────────────────────────────────────────
    year    : List[int] | int | None = None
    quarter : List[int] | int | None = None

    # ---------- NORMALISERS ----------
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
        bad   = [x for x in items if x not in ALLOWED_TYPES]
        if bad:
            raise ValueError(f"Unknown type(s): {', '.join(bad)}")
        return items

    @validator("form", pre=True)
    def _norm_form(cls, v):
        if v is None:
            return []
        items = [_norm_form(v)] if isinstance(v, str) else [_norm_form(x) for x in v]
        bad   = [x for x in items if x not in ALLOWED_FORMS]
        if bad:
            raise ValueError(f"Unknown form(s): {', '.join(bad)}")
        return items

    @field_validator("year", "quarter", mode="before")
    def _to_int_list(cls, v, info: ValidationInfo):
        """
        Accept single int/str or list[int/str]; always return List[int].
        Also range-check the quarter field.
        """
        if v is None:
            return []

        # normalise to list
        items = v if isinstance(v, list) else [v]
        ints  = [int(x) for x in items]

        if info.field_name == "quarter" and any(q not in range(1, 5) for q in ints):
            raise ValueError("quarter must be between 1 and 4")

        return ints

    # ---------- CONVERSION TO PINECONE ----------
    def to_pinecone_filter(self) -> Dict[str, Any]:
        def one_or_in(seq):
            if not seq:
                return None
            return seq[0] if len(seq) == 1 else {"$in": seq}

        return {k: v for k, v in {
            "symbol" : one_or_in(self.symbol),
            "type"   : one_or_in(self.type),
            "form"   : one_or_in(self.form),
            "year"   : one_or_in(self.year),
            "quarter": one_or_in(self.quarter),
        }.items() if v is not None}



# ------------------------- LLM extraction chain (GPT-4o) -------------------------
_extract_llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=CHAT_DEPLOYMENT_NAME,
    model_name=CHAT_MODEL,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)

_parser = PydanticOutputParser(pydantic_object=QueryFilter)

_extract_prompt = PromptTemplate(
    template="""\
**Task**  
Extract only the *metadata filters* explicitly requested in the question.

Return **_valid JSON_** – nothing else – matching this schema:
{format}

**Allowed values**
• "type"   → {{"sec_filing","earnings_call_transcript","company_profile"}}  
• "form"   → {{"10-K","10-Q"}}  
• "year"   → integers 1900-2100 (single or list)  
• "quarter"→ 1-4 (single or list)  
• "symbol" → stock tickers (single or list)

Always emit arrays when multiple values are present.

User question:
"{question}"
""",
    input_variables=["question"],
    partial_variables={"format": _parser.get_format_instructions()},
)


extract_chain = _extract_prompt | _extract_llm | _parser


# ------------------------- Reciprocal-Rank Fusion helper -------------------------
def rrf_merge(dense: List[Document], sparse: List[Document], k: int = 20) -> List[Document]:
    scores: Dict[str, float]   = {}
    seen:   Dict[str, Document] = {}

    for rank, doc in enumerate(dense):
        key = doc.metadata.get("id") or id(doc)      # ← use vector id
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        seen[key]   = doc

    for rank, doc in enumerate(sparse):
        key = doc.metadata.get("id") or id(doc)      # ← same here
        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        seen[key]   = doc

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [seen[key] for key, _ in ranked]



# ------------------------- Dynamic retrieve function factory -------------------------
def make_retrieve_fn(
    dense_store: PineconeVectorStore,
    sparse_index,
    sparse_encoder,
) -> RunnableLambda:
    """
    Returns a Runnable that
      1. extracts metadata filters from the question
      2. performs progressively-relaxed searches (dense + sparse)
      3. merges the two result lists with RRF
    """

    # ──────────────────────────────────────────────────────────────────────
    # helper nested inside so it can use dense_store / sparse_index
    # ──────────────────────────────────────────────────────────────────────
    def _sparse(q_text: str, flt: Dict[str, Any]) -> list[Document]:
        sv  = sparse_encoder.encode_queries([q_text])[0]
        res = sparse_index.query(
            sparse_vector=sv,
            top_k=TOP_K_PER_STAGE,
            include_metadata=True,
            filter=flt or None
        )
        return [
            Document(page_content=m["metadata"]["text"], metadata=m["metadata"])
            for m in res["matches"]
        ]

    def pinecone_search(q_text: str, base_filter: Dict[str, Any]) -> tuple[list[Document], list[Document]]:
        """Try full filter → drop quarter → drop year → no filter."""
        # 1. full filter
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=base_filter or None)
        if dense:
            sparse = _sparse(q_text, base_filter)
            return dense, sparse

        # 2️. drop quarter
        f2 = {k: v for k, v in base_filter.items() if k != "quarter"}
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=f2 or None)
        if dense:
            sparse = _sparse(q_text, f2)
            return dense, sparse

        # 3️. drop year
        f3 = {k: v for k, v in f2.items() if k != "year"}
        dense = dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE, filter=f3 or None)
        if dense:
            sparse = _sparse(q_text, f3)
            return dense, sparse

        # 4️. no filter
        return (
            dense_store.similarity_search(q_text, k=TOP_K_PER_STAGE),
            _sparse(q_text, {})  
        )
    
    def _retrieve(fused_query: str, convo_id: str = "default") -> List[Document]:
        """
        * fused_query   –  all user questions joined by `\n`
        * convo_id      –  stickiness per chat session
        """
        # 1️⃣ parse metadata from the *fused* 5-turn window
        qf        = extract_chain.invoke({"question": fused_query})
        fresh_flt   = qf.to_pinecone_filter()

        # 2️⃣ merge with previous sticky filter
        base_flt    = _merge_filters(LAST_FILTER.get(convo_id, {}), fresh_flt)

        # 3️⃣ run retrieval
        dense_docs, sparse_docs = pinecone_search(fused_query, base_flt)
        merged = rrf_merge(dense_docs, sparse_docs, k=TOP_K_PER_STAGE * 2)

        # 4️⃣ filing / chunk caps (unchanged)
        by_blob: Dict[str, List[Document]] = {}
        for d in merged:
            key = d.metadata.get("symbol") + "|" + d.metadata.get("blob_name")
            by_blob.setdefault(key, []).append(d)

        per_filing = [d for docs in by_blob.values() for d in docs[:CHUNKS_PER_DOC]]
        final      = per_filing[:TOTAL_RETRIEVED_DOC_CHUNKS]

        # 5️⃣ save sticky filter *only* if retrieval succeeded
        if merged:
            LAST_FILTER[convo_id] = base_flt

        return final
    
    return RunnableLambda(lambda q: _retrieve(q))

def _is_effectively_empty(f: Dict[str, Any]) -> bool:
    """
    A filter is 'empty' when it does **not** constrain retrieval.
    We treat {'symbol': [''], 'type': None, …} or {} as empty.
    """
    if not f:
        return True
    # symbol could be '', [], or {"$in": []}
    sym = f.get("symbol")
    if sym in (None, "", [], {"$in": []}):
        sym_ok = True
    else:
        sym_ok = False
    others_present = any(k for k in f if k != "symbol")
    return sym_ok and not others_present

# 1️⃣  the LLM used inside memory
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

def build_rag_chain():
    # -------------------------
    # 2 · Initialize Azure OpenAI Embeddings (dense)
    # -------------------------
    embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VER,
    )

    # -------------------------
    # 3 · Initialize Pinecone client + indexes
    # -------------------------
    pc = Pinecone(api_key=PINECONE_API_KEY)

    dense_index  = pc.Index(DENSE_INDEX_NAME)
    sparse_index = pc.Index(SPARSE_INDEX_NAME)

    # -------------------------
    # 4 · Stores & encoders
    # -------------------------
    dense_store = PineconeVectorStore(
        index=dense_index,
        embedding=embedding,
        namespace=""
    )
    sparse_encoder = SpladeEncoder()

    # -------------------------
    # 5 · Dynamic retriever (dense + sparse with metadata filter)
    # -------------------------
    retrieve_fn = make_retrieve_fn(dense_store, sparse_index, sparse_encoder)

    # -------------------------
    # 6 · LLM Setup (unchanged)
    # -------------------------
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=CHAT_DEPLOYMENT_NAME,
        model_name=CHAT_MODEL,
        api_version=AZURE_OPENAI_API_VER,
        max_tokens=1024,
        temperature=0.25
    )

    # -------------------------
    # 8 · RAG chain (context → prompt → llm)
    # -------------------------
    prompt = PromptTemplate.from_template(
        """
        You are an intelligent assistant specializing in financial & investment analysis.

        Conversation so far:
        {history}

        Guidelines:
        - Your response should be **clear and informative**.
        - Answer in **as much detail as the question requires** — use a few sentences for simple questions and multiple paragraphs for more complex ones.
        - Structure your answer with full sentences and logical flow.
        - Cite retrieved information in-text, example: (AAPL 10K 2024).
        - **Ignore advertisements** or irrelevant content in the documents.
        - Be professional and concise — avoid filler, but don’t cut important details.

        Question: {question}

        Context: {context}

        Answer:
        """
    )

    format_docs = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))
    rag_chain = (
        {
            "context": RunnableLambda(lambda _: build_fused_query()) | retrieve_fn
                    | (lambda docs: "\n\n".join(d.page_content for d in docs)),

            "question":  itemgetter("prompt_question"),

            # ⬇️ load the text buffer from memory at call-time
            "history":   RunnableLambda(lambda _: memory.load_memory_variables({})["history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Keep a global instance so it’s not re‐initialized on every request
RAG_CHAIN = build_rag_chain()


def stream_answer(question: str) -> Iterable[str]:
    """FastAPI streamer that updates user-query buffer and memory."""
    # 1⃣  cache the user question
    USER_QUERIES.append(question)

    inputs = {
        "prompt_question": question,   # fed to prompt
    }

    chunks = []
    for piece in RAG_CHAIN.stream(inputs):
        text = piece if isinstance(piece, str) else piece.content
        chunks.append(text)
        yield text

    # 2⃣  store turn in memory
    memory.save_context({"input": question},
                        {"output": "".join(chunks)})
