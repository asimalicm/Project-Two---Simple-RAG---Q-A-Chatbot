import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

class DocumentQA:

    def __init__(self):

        self.is_document_loaded = False
        self.vector_store = None
        self.retriever = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

        self.prompt_template = PromptTemplate(
            template="""You are a precise assistant that answers questions \
using ONLY the document context provided below.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the context.
2. If the answer is not in the context, respond with exactly:
   "I cannot find this information in the document."
3. Do NOT use your own general knowledge to fill in any gaps.
4. Be concise. Quote the document directly where helpful.
5. If the context partially answers the question, share what you
   found and note what is missing.

--- DOCUMENT CONTEXT ---
{context}
--- END OF CONTEXT ---

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        print("RAG system initialised.")
        print("  Text splitter  → chunk_size=1000, overlap=200")
        print("  Embeddings     → gemini-embedding-001 (Gemini)")
        print("  Vector store   → InMemoryVectorStore (built-in LangChain)")
        print("  Language model → gemini-2.5-flash (temperature=0)")
        print()

    def load_document(self, text: str) -> None:

        if not text.strip():
            print("Cannot load an empty document.")
            return

        print(f"\n[Step 1/3] Document received  ({len(text):,} characters)")

        raw_documents = [Document(
            page_content=text,
            metadata={"source": "user_input"}
        )]
        chunks = self.text_splitter.split_documents(raw_documents)

        print(f"[Step 1/3] Split into {len(chunks)} chunk(s):")
        for i, chunk in enumerate(chunks, 1):
            print(f"           Chunk {i}: {len(chunk.page_content)} characters")

        print(f"[Step 2/3] Embedding chunks and storing in memory...")
        print(f"           (Calls Gemini API once per chunk — takes a moment)")

        self.vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        print(f"           {len(chunks)} chunk(s) embedded and stored.")

        print(f"[Step 3/3] Building retriever (k=3)...")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.is_document_loaded = True
        print()
        print("Document loaded! You can now ask questions.")
        print()

    def ask_question(self, question: str) -> None:

        if not self.is_document_loaded:
            print()
            print("No document loaded yet. Use /load to add one first.")
            print()
            return

        if not question.strip():
            print("Please type a question.")
            return

        print()
        print(f'Searching for: "{question}"')

        try:
            print("Retrieving relevant chunks from vector store...")
            source_docs = self.retriever.invoke(question)

            if not source_docs:
                print("No relevant chunks found. The document may be too short.")
                return

            context = "\n\n---\n\n".join(
                doc.page_content for doc in source_docs
            )

            filled_prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            print("Generating answer with Gemini...")
            response = self.llm.invoke([HumanMessage(content=filled_prompt)])

            answer = response.content

            print()
            print("=" * 55)
            print("  ANSWER")
            print("=" * 55)
            print(answer)

            print()
            print("-" * 55)
            print(f"  SOURCES  ({len(source_docs)} chunk(s) retrieved)")
            print("-" * 55)
            for i, doc in enumerate(source_docs, 1):
                preview = doc.page_content[:220].replace("\n", " ").strip()
                if len(doc.page_content) > 220:
                    preview += "..."
                print(f"  [Chunk {i}]: {preview}")
                print()
            print("=" * 55)
            print()

        except Exception as error:
            print()
            print(f"Error: {error}")
            print("Check your GEMINI_API_KEY and internet connection.")
            print()


def display_welcome_banner() -> None:

    print()
    print("=" * 55)
    print("  PROJECT 2 — Smart Document Q&A Bot (RAG)")
    print("=" * 55)
    print()
    print("  HOW IT WORKS:")
    print("  1. Use /load to paste in a document")
    print("  2. Type any question about the document")
    print("  3. The AI answers using ONLY that document")
    print("  4. You'll see which sections it pulled from")
    print()
    print("  COMMANDS:")
    print("  /load     → Paste a document (end with ###END###)")
    print("  /quit     → Exit")
    print("  anything  → Ask a question about the document")
    print()
    print("=" * 55)
    print()


def collect_multiline_input() -> str:

    print()
    print("Paste your document text below.")
    print("When done, type  ###END###  on a new line and press Enter:")
    print("-" * 55)

    collected_lines = []

    while True:
        try:
            line = input()

            if line.strip() == "###END###":
                break

            collected_lines.append(line)

        except EOFError:
            break

    return "\n".join(collected_lines)


def main() -> None:

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print()
        print("ERROR: GEMINI_API_KEY not found.")
        print("Fix:")
        print("  1. Copy .env.example to .env")
        print("  2. Edit .env: GEMINI_API_KEY=your_key_here")
        print("  3. Get a key at: https://aistudio.google.com/app/apikey")
        print()
        sys.exit(1)

    display_welcome_banner()

    qa_system = DocumentQA()

    print("Ready. Type /load to add a document, or /quit to exit.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            print("Goodbye!")
            print()
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print()
            print("Goodbye! Keep building.")
            print()
            break

        elif user_input.lower() == "/load":
            document_text = collect_multiline_input()

            if document_text.strip():
                qa_system.load_document(document_text)
            else:
                print()
                print("No text was entered. Document not loaded.")
                print()

        else:
            qa_system.ask_question(user_input)


if __name__ == "__main__":
    main()