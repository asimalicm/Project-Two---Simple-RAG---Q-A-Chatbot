# ============================================================
#  PROJECT 2: Smart Document Q&A Bot (RAG System)
# ============================================================
#
#  WHAT IS RAG? (Retrieval Augmented Generation)
#  -----------------------------------------------
#  Normal AI chat:  You ask → AI answers from its training memory
#  RAG chat:        You give it a doc → You ask → AI answers from YOUR doc
#
#  Think of it like the difference between:
#  - Closed-book exam (normal AI): relies only on memorized knowledge
#  - Open-book exam (RAG):         looks up the relevant page first, then answers
#
#  THE 4-STEP RAG PIPELINE:
#  ┌─────────────────────────────────────────────────────┐
#  │  1. CHUNK   → Split big document into small pieces  │
#  │  2. EMBED   → Convert each piece into a number list │
#  │  3. STORE   → Save those number lists in memory     │
#  │  4. ANSWER  → On question, find closest pieces,     │
#  │               feed them to AI, get an answer        │
#  └─────────────────────────────────────────────────────┘
#
#  ARCHITECTURE NOTE (Python 3.14 + Modern LangChain 1.x)
#  -------------------------------------------------------
#  LangChain 1.x removed the old "RetrievalQA" chain.
#  We now build the pipeline explicitly — which is actually
#  MORE educational because you can see every step clearly:
#    question → retrieve chunks → format prompt → LLM → answer
#
#  HOW TO RUN:
#  -----------
#  1. pip install -r requirements.txt
#  2. Create .env file with GEMINI_API_KEY=your_key
#  3. python rag_chatbot.py
#  4. Type /load, paste your text, type ###END###
#  5. Ask questions!
# ============================================================

import os
import sys
import warnings

# Suppress the Pydantic V1 deprecation warning from langchain_core internals.
# This warning appears because langchain_core still touches pydantic.v1 for
# backward compatibility, but it doesn't affect our code at all.
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# python-dotenv reads your .env file and loads GEMINI_API_KEY
# into os.environ so os.getenv() can access it safely
from dotenv import load_dotenv

# ---- LangChain imports (modern, LangChain 1.x compatible) ----
# LangChain is a framework for building AI pipelines.
# In version 1.x, the core building blocks moved to langchain_core.

# Splits a big string into overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini: the LLM that generates answers (reads context → writes answer)
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini: the embedding model that converts text → list of numbers
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# InMemoryVectorStore: stores embeddings in RAM for fast similarity search.
# Same concept as ChromaDB but with zero external dependencies.
# ChromaDB currently has a compatibility issue with Python 3.14 (pydantic.v1
# compat layer breaks). InMemoryVectorStore teaches the identical RAG concepts.
#
# TO SWAP IN CHROMADB when it supports Python 3.14:
#   pip install langchain-chroma chromadb
#   from langchain_chroma import Chroma
#   self.vector_store = Chroma.from_documents(chunks, embedding=self.embeddings,
#                                              collection_name="rag_doc")
from langchain_core.vectorstores import InMemoryVectorStore

# PromptTemplate: defines a reusable prompt with {placeholders}
# HumanMessage: wraps a string as a "user turn" message for the chat model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Document: a LangChain container for text + metadata
# metadata = extra info like {"source": "page 3", "author": "Asim"}
# In LangChain 1.x, Document moved from langchain.schema to langchain_core.documents
from langchain_core.documents import Document


# ============================================================
#  THE MAIN CLASS: DocumentQA
# ============================================================

class DocumentQA:
    """
    Encapsulates the entire RAG pipeline in one reusable object.

    WHY a class?
    ------------
    A class lets us "hold onto" state between method calls.
    We need to keep:
      - self.vector_store → the in-memory database with our embeddings
      - self.retriever    → the search interface into the vector store
      - self.prompt       → the anti-hallucination prompt template

    Without a class, we'd need globals or re-process the document
    before every single question.

    Analogy: The class is a filing cabinet. You load papers once
    (load_document), then pull out relevant ones per question (ask_question).
    """

    def __init__(self):
        """
        Creates all the reusable components that don't change between
        documents: text splitter, embeddings model, LLM, and prompt.

        These are created ONCE here rather than inside load_document
        because they don't depend on any specific document.
        Initialising them once saves time and API calls.
        """

        # --- State flags ---
        self.is_document_loaded = False
        self.vector_store = None
        self.retriever = None

        # ---- 1. TEXT SPLITTER ----
        # WHY do we split text into chunks?
        # ─────────────────────────────────
        # AI models have a "context window" — a hard cap on how much
        # text they can process at once (~30,000 characters for Flash).
        # A 50-page document is too large. We need to:
        #   a) Split the doc into small pieces
        #   b) Find ONLY the 2-3 pieces relevant to the question
        #   c) Send just those pieces to the AI
        #
        # chunk_size=1000:  Each chunk is ~1000 characters (~150 words)
        # chunk_overlap=200: Neighbouring chunks share 200 characters
        #
        # WHY overlap?
        # Without it, a sentence that falls exactly on a boundary gets
        # cut in half. With overlap, both neighbouring chunks contain
        # that sentence, so no information is ever lost at the seams.
        # Think of it like overlapping shingles on a roof — no gaps.
        #
        # separators=["\n\n", "\n", " ", ""]:
        # Tries to split at paragraph breaks first (most natural),
        # then line breaks, then word spaces, then individual characters.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # ---- 2. EMBEDDINGS MODEL ----
        # WHAT are embeddings? (The KEY concept in RAG)
        # ──────────────────────────────────────────────
        # An embedding converts text into a list of ~768 numbers (a "vector").
        # Texts with similar MEANINGS get similar vectors:
        #   "dog"   → [0.12, -0.45, 0.88, ...]
        #   "puppy" → [0.11, -0.44, 0.87, ...]  ← very close!
        #   "rocket"→ [-0.91, 0.33, -0.12, ...]  ← very different
        #
        # WHY does this matter for RAG?
        # When you ask "What is the refund period?", we embed that question
        # and search for the document chunks with the most SIMILAR vectors.
        # This finds chunks about refunds even if they say "return window"
        # or "money-back guarantee" — different words, same meaning.
        # This is called SEMANTIC SEARCH and beats keyword search completely.
        # "models/embedding-001" was retired in the newer google-genai SDK.
        # "models/text-embedding-004" is the current recommended model.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # ---- 3. LANGUAGE MODEL ----
        # This is the AI that READS the retrieved chunks and WRITES the answer.
        # It only runs AFTER the relevant chunks have been found.
        #
        # temperature=0: Completely deterministic output.
        # Temperature controls "creativity" / randomness:
        #   0.0 = always picks the highest-probability next word (factual)
        #   1.0 = sometimes picks surprising words (creative, varied)
        # For Q&A we want accuracy, so temperature=0.
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

        # ---- 4. PROMPT TEMPLATE ----
        # This is the most important part for PREVENTING HALLUCINATION.
        #
        # Without a strict prompt, Gemini might fill in gaps from its
        # training data ("I know CEOs are usually named X...").
        # Our prompt EXPLICITLY instructs it:
        #   - Use ONLY the provided context
        #   - Say "I cannot find this information" if it's not there
        #
        # {context} → filled with the retrieved document chunks
        # {question} → filled with the user's question
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
        print("  Embeddings     → models/text-embedding-001 (Gemini)")
        print("  Vector store   → InMemoryVectorStore (built-in LangChain)")
        print("  Language model → gemini-2.5-flash (temperature=0)")
        print()

    # ----------------------------------------------------------

    def load_document(self, text: str) -> None:
        """
        Takes raw text, runs it through the full preparation pipeline,
        and stores it ready to be queried.

        Pipeline (explicit, step-by-step):
          raw text
            → Document objects (text + metadata wrapper)
            → chunks  (split by RecursiveCharacterTextSplitter)
            → vectors (embed each chunk via Gemini)
            → store   (InMemoryVectorStore holds text + vector pairs)
            → retriever (search interface, configured with k=3)
            → ready to answer questions!

        Parameters:
            text (str): The raw document text the user pasted in.
        """

        if not text.strip():
            print("Cannot load an empty document.")
            return

        print(f"\n[Step 1/3] Document received  ({len(text):,} characters)")

        # ── STEP 1: WRAP TEXT + SPLIT INTO CHUNKS ───────────────────
        # LangChain's Document object is a simple container:
        #   .page_content  → the actual text
        #   .metadata      → a dict of extra info (source, page, date…)
        #
        # split_documents() copies metadata to every child chunk, so
        # we never lose track of where a chunk came from.
        raw_documents = [Document(
            page_content=text,
            metadata={"source": "user_input"}
        )]
        chunks = self.text_splitter.split_documents(raw_documents)

        print(f"[Step 1/3] Split into {len(chunks)} chunk(s):")
        for i, chunk in enumerate(chunks, 1):
            print(f"           Chunk {i}: {len(chunk.page_content)} characters")

        # ── STEP 2: EMBED CHUNKS + STORE ────────────────────────────
        # from_documents() does these things automatically:
        #   a) Sends each chunk's text to the Gemini embedding model
        #   b) Gets back a vector (list of ~768 floats)
        #   c) Stores the (text, vector, metadata) triple in RAM
        #
        # After this, the store can answer:
        # "give me the 3 stored vectors most similar to THIS query vector"
        print(f"[Step 2/3] Embedding chunks and storing in memory...")
        print(f"           (Calls Gemini API once per chunk — takes a moment)")

        self.vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        print(f"           {len(chunks)} chunk(s) embedded and stored.")

        # ── STEP 3: BUILD THE RETRIEVER ──────────────────────────────
        # The retriever is the search interface over the vector store.
        # as_retriever() wraps the store so we can call:
        #   retriever.invoke("some question")  → list of Document objects
        #
        # search_type="similarity" → find vectors closest in direction
        # k=3 → return the 3 most relevant chunks
        #
        # Why 3?
        #   Too few (k=1) → might miss context split across chunks
        #   Too many (k=10) → floods the prompt with noise, costs more
        print(f"[Step 3/3] Building retriever (k=3)...")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.is_document_loaded = True
        print()
        print("Document loaded! You can now ask questions.")
        print()

    # ----------------------------------------------------------

    def ask_question(self, question: str) -> None:
        """
        Runs a question through the full RAG pipeline step-by-step
        and prints:
          1. The AI's answer
          2. The document chunks that were used to produce it

        WHY show the source chunks?
        That's a core RAG principle called GROUNDING — you can verify
        the AI's answer against the exact text it used. If it's wrong,
        you can check whether the right chunks were even retrieved.

        The pipeline (all manual and visible):
          question → retriever → top-3 chunks → format prompt → LLM → answer

        Parameters:
            question (str): The user's natural language question.
        """

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
            # ── STEP A: RETRIEVE RELEVANT CHUNKS ────────────────────
            # retriever.invoke() does:
            #   1. Embeds the question using the same Gemini model
            #   2. Compares the question vector to all stored chunk vectors
            #   3. Returns the top k=3 chunks with the most similar vectors
            #
            # This is the heart of RAG — semantic search over your document.
            print("Retrieving relevant chunks from vector store...")
            source_docs = self.retriever.invoke(question)

            if not source_docs:
                print("No relevant chunks found. The document may be too short.")
                return

            # ── STEP B: FORMAT THE CONTEXT ───────────────────────────
            # Join all retrieved chunks into one block of text.
            # The "---" separator makes it clear to the LLM where one
            # chunk ends and another begins.
            context = "\n\n---\n\n".join(
                doc.page_content for doc in source_docs
            )

            # ── STEP C: BUILD THE FULL PROMPT ────────────────────────
            # Fill in the {context} and {question} placeholders.
            # .format() is standard Python string formatting.
            filled_prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            # ── STEP D: SEND TO THE LLM ──────────────────────────────
            # We wrap the prompt as a HumanMessage because ChatGoogleGenerativeAI
            # is a chat model — it expects conversation turns, not raw strings.
            # HumanMessage = a message from the "human" side of the conversation.
            print("Generating answer with Gemini...")
            response = self.llm.invoke([HumanMessage(content=filled_prompt)])

            # .content extracts the text from the AIMessage response object
            answer = response.content

            # ── DISPLAY THE ANSWER ───────────────────────────────────
            print()
            print("=" * 55)
            print("  ANSWER")
            print("=" * 55)
            print(answer)

            # ── DISPLAY THE SOURCE CHUNKS ────────────────────────────
            # Show exactly which document sections the answer came from.
            # This is RAG transparency — you can audit every answer.
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


# ============================================================
#  HELPER FUNCTIONS (CLI / UX)
# ============================================================

def display_welcome_banner() -> None:
    """
    Prints a helpful welcome screen when the program starts.
    Pure UI logic — no AI here.
    """
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
    """
    Reads multi-line text from the terminal until the user types ###END###.

    WHY do we need this?
    --------------------
    Python's input() reads exactly ONE line at a time. Documents are
    multi-line, so we need a loop that keeps collecting lines until a
    special "stop signal" is entered.

    ###END### is the stop signal — easy to type, unlikely to appear in
    real document text.

    Returns:
        str: All lines joined with newlines (not including ###END###).
    """
    print()
    print("Paste your document text below.")
    print("When done, type  ###END###  on a new line and press Enter:")
    print("-" * 55)

    collected_lines = []

    while True:
        try:
            line = input()

            # Stop collecting when the sentinel line is detected
            if line.strip() == "###END###":
                break

            collected_lines.append(line)

        except EOFError:
            # Handles Ctrl+D gracefully (common on Linux/Mac)
            break

    return "\n".join(collected_lines)


# ============================================================
#  ENTRY POINT: main()
# ============================================================

def main() -> None:
    """
    Sets up the chatbot and runs the interactive command loop.

    This implements the classic REPL pattern:
      Read → Evaluate → Print → Loop

    Steps:
      1. Load .env and verify the API key exists
      2. Print the welcome banner
      3. Initialise the DocumentQA system
      4. Loop forever, routing input to the right handler
    """

    # Load .env — MUST happen before os.getenv() calls
    load_dotenv()

    # Fail early and clearly if the API key is missing.
    # Better to crash here with a helpful message than get a confusing
    # "403 Forbidden" error deep inside a LangChain call.
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

    # Initialise the RAG system — creates the splitter, embeddings model,
    # LLM and prompt template. Does NOT load any document yet.
    qa_system = DocumentQA()

    print("Ready. Type /load to add a document, or /quit to exit.")
    print()

    # ── MAIN COMMAND LOOP ────────────────────────────────────────────
    # Runs until the user types /quit or presses Ctrl+C.
    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D — exit gracefully
            print()
            print("Goodbye!")
            print()
            break

        # Skip blank inputs
        if not user_input:
            continue

        # ── ROUTE THE INPUT ──────────────────────────────────────────
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
            # Any input that isn't a command is treated as a question
            qa_system.ask_question(user_input)


# ── STANDARD PYTHON ENTRY POINT GUARD ───────────────────────────────
# WHY this pattern?
# When Python IMPORTS a file it runs all top-level code.
# This guard means main() ONLY runs when you execute the file directly:
#   python rag_chatbot.py          ← main() runs
#   from rag_chatbot import DocumentQA  ← main() does NOT run
# This makes code both runnable AND importable as a module.
if __name__ == "__main__":
    main()
