# 🤖 Smart Document Q&A Bot (RAG)

A simple, educational Python project that lets you **chat with your own documents**. 

This is a **RAG (Retrieval Augmented Generation)** system built from scratch. It doesn't just "know" things from its training; it looks up answers in the text you provide, making it perfect for summarizing articles, asking questions about specific reports, or just learning how modern AI systems work.

## 🌟 Why this project?

If you're learning AI Engineering, **RAG** is one of the most important concepts to master. This project strips away the complexity of big vector databases and cloud infrastructure to show you the *core mechanics* of how AI reads and retrieves information.

**You will learn:**
- How to **split** text into manageable chunks.
- How **embeddings** turn text into numbers (vectors).
- How **semantic search** finds relevant information.
- How to prevent AI **hallucinations** by grounding answers in facts.

## ✨ Features

- **Bring Your Own Data**: Paste any text document, and the bot will learn it instantly.
- **Powered by Gemini**: Uses Google's latest Gemini 2.5 Flash model for fast, accurate answers.
- **Transparent**: Shows you exactly which parts of the document it used to answer your question.
- **Zero Database Setup**: Uses an in-memory vector store, so there's no complex database to install.
- **Heavily Commented Code**: Every function and class is explained like a tutorial.

## 🛠️ Prerequisites

- **Python 3.10+** installed on your machine.
- A **Google Cloud API Key** (free tier available).

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API Key
Create a `.env` file in the project root and add your Google Gemini API key:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```
*(You can get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 4. Run the bot
```bash
python rag_chatbot.py
```

## 🎮 How to Use

1. **Start the bot**: You'll see a welcome message.
2. **Load a document**: Type `/load` and paste your text. Type `###END###` on a new line when finished.
3. **Ask questions**: Type anything!
   - *"What is the main conclusion?"*
   - *"Summarize the safety protocols."*
   - *"Who is the author?"*
4. **See the magic**: The bot will answer and show you the **Sources** (the exact text chunks it found).

## 🧠 How it Works (The "Magic")

This project implements a standard **RAG Pipeline**:

1.  **Chunking**: Your long document is sliced into smaller, overlapping pieces (chunks).
2.  **Embedding**: Each chunk is converted into a list of numbers (a vector) that represents its *meaning*.
3.  **Storage**: These vectors are saved in a temporary in-memory database.
4.  **Retrieval**: When you ask a question, the system converts your question into numbers and finds the chunks that are mathematically closest (most similar) to it.
5.  **Generation**: The system sends your question + the found chunks to the AI and says, *"Using ONLY these chunks, answer the question."*

## 📂 Project Structure

- `rag_chatbot.py`: The main application code.
- `rag_chatbot_learn.py`: A version with extra comments for learning.
- `.env`: Stores your private API key (do not share this!).

## 🤝 Contributing

Feel free to fork this project and add features like PDF loading, a web interface (Streamlit), or different LLM providers!

---
*Happy Coding!* 🚀
