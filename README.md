# LangChain-Based-News-Research-System
RAG-based news research tool using LangChain and FAISS vector database
A RAG (Retrieval-Augmented Generation) powered application that:

1. Scrapes news articles from URLs
2. Indexes them in a searchable vector database
3. Answers questions about the articles with source attribution
4. Provides accurate, context-aware responses based on actual article content (not hallucinations)

How It Works:
1. Data Ingestion (Article Scraping)
URLs → Scraper → Raw Text → Text Chunks

Takes news article URLs as input
Extracts article content (headline, body, metadata)
Splits long articles into smaller chunks for processing

2. Vector Embedding & Storage
Text Chunks → OpenAI Embeddings → FAISS Vector Database

Converts text chunks into numerical vectors (embeddings)
Stores vectors in FAISS (Facebook AI Similarity Search) for fast retrieval
Each chunk maintains link to original source

3. Query & Retrieval
User Question → Embedding → FAISS Search → Relevant Chunks

User asks a question about the news
Question is converted to embedding
FAISS finds most similar/relevant article chunks
Retrieved chunks provide context

4. Answer Generation
Question + Retrieved Context → LangChain + OpenAI → Answer with Sources

LangChain orchestrates the workflow
Sends question + relevant chunks to LLM (GPT)
LLM generates answer grounded in retrieved content
System includes source attribution (which articles were used)
