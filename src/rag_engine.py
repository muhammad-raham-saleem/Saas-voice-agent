import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


class RAGEngine:
    """RAG engine for TechFlow AI customer support."""

    def __init__(self, knowledge_base_dir: str = None, vectorstore_dir: str = None):
        self.knowledge_base_dir = Path(knowledge_base_dir) if knowledge_base_dir else KNOWLEDGE_BASE_DIR
        self.vectorstore_dir = Path(vectorstore_dir) if vectorstore_dir else VECTORSTORE_DIR

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in environment variables. Add it to your .env file.")

        print("✅ OpenAI API key loaded")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.doc_count = 0
        self.chunk_count = 0

        self._initialize()

    def _initialize(self):
        """Load knowledge base, build vector store, and create the chain."""
        if self.vectorstore_dir.exists():
            print("📂 Found existing vector store, loading...")
            self.load_vectorstore()
        else:
            print("📚 Building vector store from knowledge base...")
            self._build_vectorstore()

        self._build_chain()
        print("✅ RAG engine ready!")

    def _build_vectorstore(self):
        """Load documents, split into chunks, and create FAISS vector store."""
        loader = DirectoryLoader(
            str(self.knowledge_base_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        documents = loader.load()
        self.doc_count = len(documents)
        print(f"  📄 Loaded {self.doc_count} documents")

        chunks = self.text_splitter.split_documents(documents)
        self.chunk_count = len(chunks)
        print(f"  🔪 Split into {self.chunk_count} chunks")

        print("  🧠 Creating embeddings (this may take a moment)...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(f"  ✅ FAISS vector store created with {self.chunk_count} vectors")

        self.save_vectorstore()

    def _build_chain(self):
        """Build the conversational retrieval chain."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a friendly and professional customer support agent for TechFlow, a project management software company. Use the following context to answer the customer's question.

Rules:
- Only answer based on the provided context. Do not make up information.
- If the context does not contain enough information to answer the question, say: "I don't have information about that in our knowledge base. Please contact our support team at support@techflow.io or call +1-888-TECHFLOW for further assistance."
- Be concise, helpful, and professional.
- If the customer seems frustrated, be empathetic.
- Format your response with bullet points or numbered steps when listing multiple items.

Context:
{context}

Customer Question: {question}

Answer:"""
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False,
        )
        print("  🔗 Conversational retrieval chain created")

    def query(self, question: str) -> dict:
        """Query the RAG engine and return answer with sources.

        Args:
            question: The user's question.

        Returns:
            dict with 'answer' (str) and 'sources' (list of dicts with
            'content' and 'source' keys).
        """
        start_time = time.time()

        result = self.chain.invoke({"question": question})

        elapsed = time.time() - start_time

        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            source_file = Path(doc.metadata.get("source", "unknown")).name
            snippet = doc.page_content[:200].strip()
            key = (source_file, snippet)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source_file,
                    "content": doc.page_content,
                })

        return {
            "answer": result["answer"],
            "sources": sources,
            "response_time": round(elapsed, 2),
        }

    def save_vectorstore(self):
        """Save the FAISS vector store to disk."""
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.vectorstore_dir))
        print(f"  💾 Vector store saved to {self.vectorstore_dir}")

    def load_vectorstore(self):
        """Load the FAISS vector store from disk."""
        self.vectorstore = FAISS.load_local(
            str(self.vectorstore_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        index = self.vectorstore.index
        self.chunk_count = index.ntotal
        print(f"  ✅ Loaded vector store with {self.chunk_count} vectors")

    def reset(self):
        """Rebuild the vector store from scratch and clear conversation memory."""
        print("🔄 Resetting RAG engine...")
        self.memory.clear()
        self._build_vectorstore()
        self._build_chain()
        print("✅ RAG engine reset complete!")

    def get_stats(self) -> dict:
        """Return engine statistics for the UI."""
        index = self.vectorstore.index
        return {
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "vector_dimensions": index.d,
            "total_vectors": index.ntotal,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  TechFlow RAG Engine - Standalone Test")
    print("=" * 60)

    engine = RAGEngine()

    test_questions = [
        "How do I reset my password?",
        "What's included in the Pro plan?",
        "What are the API rate limits?",
    ]

    for q in test_questions:
        print(f"\n{'─' * 50}")
        print(f"❓ {q}")
        result = engine.query(q)
        print(f"💬 {result['answer']}")
        print(f"⏱️  Response time: {result['response_time']}s")
        print(f"📎 Sources: {', '.join(s['source'] for s in result['sources'])}")
