import os, re, json, functools
from dotenv import load_dotenv
from operator import itemgetter
from typing import Iterable, List, Any, Dict, Literal, Optional
import logging

# Pydantic
from pydantic import BaseModel, Field, validator

# LangChain core
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema import Document

# LangChain prompts
from langchain.prompts import PromptTemplate

# LangChain community transformers (for duplicate filtering)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain retrievers (for ensembling)
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# Azure OpenAI embeddings & chat
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Pinecone + SPLADE
from pinecone import Pinecone              
from langchain_pinecone import PineconeVectorStore    
from pinecone_text.sparse import SpladeEncoder
from langchain_core.retrievers import BaseRetriever

# LLM cache
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


# Load environment variables once at module import
load_dotenv()

# Memory Cache for LLM
set_llm_cache(InMemoryCache())

# Config
AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
CHAT_MODEL            = os.getenv("CHAT_DEPLOYMENT_NAME")

PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
DENSE_INDEX_NAME      = os.getenv("PINECONE_DENSE_INDEX_NAME")
SPARSE_INDEX_NAME     = os.getenv("PINECONE_SPARSE_INDEX_NAME")

# ── retrieval hyper-params (env-configurable) ────────────────────────────
TOTAL_RETRIEVED_DOC_CHUNKS = int(os.getenv("TOTAL_RETRIEVED_DOC_CHUNKS", "20"))
CHUNKS_PER_DOC             = int(os.getenv("CHUNKS_PER_DOC", "5"))

# how many raw candidates to ask Pinecone for (dense & sparse)
# rule of thumb: at least 2× the final budget
TOP_K_PER_STAGE            = max(6, TOTAL_RETRIEVED_DOC_CHUNKS * 2)

logging.info("Retriever config: TOP_K_PER_STAGE=%d  PER_DOC=%d  FINAL=%d",
             TOP_K_PER_STAGE, CHUNKS_PER_DOC, TOTAL_RETRIEVED_DOC_CHUNKS)

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

class QueryFilter(BaseModel):
    """Structured representation of the user's request."""
    symbol: List[str] | str                                            
    type: Optional[Literal[
        "sec_filing", "earnings_call_transcript", "company_profile"
    ]] = None
    form: Optional[Literal["10-K", "10-Q"]] = None  
    year: Optional[int] = Field(None, ge=1900, le=2100)
    quarter: Optional[int] = Field(None, ge=1, le=4)

    @validator("symbol", pre=True, always=True)
    def upper_symbol(cls, v):
        if isinstance(v, str):
            return [v.upper()]
        # list of tickers
        return [s.upper() for s in v]

    @validator("form", pre=True)
    def normalise_form(cls, v):
        return _norm_form(v)

    def to_pinecone_filter(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        f["symbol"] = self.symbol
        if len(self.symbol) == 1:
            f["symbol"] = self.symbol[0]
        else:
            f["symbol"] = {"$in": self.symbol} 
        if self.type:   f["type"]   = self.type
        if self.form:   f["form"]   = self.form         
        if self.year is not None:    f["year"]    = int(self.year)
        if self.quarter is not None: f["quarter"] = int(self.quarter)
        return f

# ------------------------- LLM extraction chain (GPT-4o) -------------------------
_extract_llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("CHAT_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)

_parser = PydanticOutputParser(pydantic_object=QueryFilter)

_extract_prompt = PromptTemplate(
    template="""
You are a parser that extracts structured filters from a user question
about financial documents. 
Output MUST be valid JSON matching this schema:
{format}

User question: "{question}"

"You may output multiple symbols separated by commas, e.g. \"AAPL, MSFT\".\n"
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

    # ──────────────────────────────────────────────────────────────────────
    # main callable used by LangChain
    # ──────────────────────────────────────────────────────────────────────
    def _retrieve(question: str) -> List[Document]:
        qf: QueryFilter = extract_chain.invoke({"question": question})
        pc_filter       = qf.to_pinecone_filter()

        dense_docs, sparse_docs = pinecone_search(question, pc_filter)
        merged = rrf_merge(dense_docs, sparse_docs, k=TOP_K_PER_STAGE * 2)

        by_blob: Dict[str, List[Document]] = {}
        for d in merged:
            key = d.metadata.get("symbol") + "|" + d.metadata.get("blob_name")
            by_blob.setdefault(key, []).append(d)

        # flatten with a two-level cap: first per filing, then per symbol
        per_filing = [d for docs in by_blob.values() for d in docs[:CHUNKS_PER_DOC]]

        by_symbol: Dict[str, List[Document]] = {}
        for d in per_filing:
            by_symbol.setdefault(d.metadata.get("symbol", "UNK"), []).append(d)
            
        merged = (
            [d for docs in by_blob.values() for d in docs[:CHUNKS_PER_DOC]]
            [:TOTAL_RETRIEVED_DOC_CHUNKS]           # overall limit
            )
        if not pc_filter:
            debug = "[DEBUG] ⚠️  No metadata filters detected — vector-only search.\n"
            merged.insert(0, Document(page_content=debug, metadata={}))

        return merged

    return RunnableLambda(_retrieve)



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
        deployment_name=CHAT_MODEL,
        api_version=AZURE_OPENAI_API_VER,
        temperature=0.25
    )

    # -------------------------
    # 7 · Summary chain (unchanged)
    # -------------------------
    summary_text = ""

    summariser_prompt = PromptTemplate.from_template(
        "Progressively summarize the conversation.\n\n"
        "Current summary:\n{summary}\n\n"
        "New lines:\n{lines}\n\n"
        "Updated summary:"
    )
    summariser_chain = summariser_prompt | llm | StrOutputParser()

    def update_summary(lines: str) -> str:
        nonlocal summary_text
        summary_text = summariser_chain.invoke({
            "summary": summary_text,
            "lines": lines
        })
        return summary_text

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
            "context": itemgetter("retrieval_question") | retrieve_fn | format_docs,
            "question": itemgetter("prompt_question"),
            "history":  itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Keep a global instance so it’s not re‐initialized on every request
RAG_CHAIN = build_rag_chain()


def stream_answer(question: str, history_summary: str = "(no prior context)") -> Iterable[str]:
    """Generator that yields chunks for FastAPI’s StreamingResponse."""
    inputs = {
        "retrieval_question": question,
        "prompt_question":    question,
        "history":            history_summary,
    }
    for chunk in RAG_CHAIN.stream(inputs):
        yield chunk if isinstance(chunk, str) else chunk.content

