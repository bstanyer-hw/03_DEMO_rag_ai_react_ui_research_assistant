import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import Iterable, List, Any

# LangChain core
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
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


# Load environment variables once at module import
load_dotenv()

# (Optional) LangSmith tracing
if os.getenv("LANGSMITH_TRACING_ACTIVE", "false").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def build_rag_chain():
    # -------------------------
    # 1 · Azure OpenAI / Pinecone ENV
    # -------------------------
    AZURE_OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION")
    EMBEDDING_MODEL       = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    CHAT_MODEL            = os.getenv("CHAT_DEPLOYMENT_NAME")

    PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
    DENSE_INDEX_NAME      = os.getenv("PINECONE_DENSE_INDEX_NAME")
    SPARSE_INDEX_NAME     = os.getenv("PINECONE_SPARSE_INDEX_NAME")

    # -------------------------
    # 2 · Initialize Azure OpenAI Embeddings (dense)
    # -------------------------
    embedding = AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,              # e.g. "text-embedding-3-small"
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
    # 4 · Dense retriever (semantic)
    # -------------------------
    dense_store     = PineconeVectorStore(index=dense_index, embedding=embedding, namespace="sec-filings")
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 10})

    # -------------------------
    # 5 · Sparse retriever (SPLADE)
    # -------------------------
    sparse_encoder = SpladeEncoder()

    class PineconeSparseRetriever(BaseRetriever):
        """Query a SPLADE‐only Pinecone index and return LangChain Documents."""
        index: Any     # a pinecone.Index
        encoder: Any   # a SpladeEncoder
        top_k: int = 10

        def _get_relevant_documents(
            self, query: str, *, callbacks=None, **kwargs
        ) -> List[Document]:
            # 1. Encode the query into a sparse vector via SPLADE
            sparse_vector = self.encoder.encode_queries([query])[0]
            # 2. Query the Pinecone sparse index
            result = self.index.query(
                top_k=self.top_k,
                sparse_vector=sparse_vector,
                include_metadata=True
            )
            # 3. Convert matches into LangChain Documents
            docs: List[Document] = []
            for match in result["matches"]:
                docs.append(
                    Document(
                        page_content=match["metadata"]["text"],
                        metadata=match["metadata"]
                    )
                )
            return docs

    sparse_retriever = PineconeSparseRetriever(
        index=sparse_index,
        encoder=sparse_encoder,
        top_k=10
    )

    # -------------------------
    # 6 · Ensemble retriever (dense + sparse only)
    # -------------------------
    base_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.75, 0.25]
    )

    # Redundancy filter (remove near-duplicates)
    deduper = EmbeddingsRedundantFilter(
        embeddings=embedding,
        similarity_threshold=0.90
    )
    retriever = base_retriever | RunnableLambda(
        lambda docs: deduper.transform_documents(docs)[:25]
    )

    # -------------------------
    # 7 · LLM Setup (unchanged)
    # -------------------------
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=CHAT_MODEL,
        api_version=AZURE_OPENAI_API_VER,
        temperature=0.25
    )

    # -------------------------
    # 8 · Summary chain (unchanged)
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
    # 9 · RAG chain (unchanged)
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
            "context": itemgetter("retrieval_question") | retriever | format_docs,
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
