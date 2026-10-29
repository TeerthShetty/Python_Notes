# Tryingto run local LLama model on my hardware
# GPU : NVIDIA RTX 3050 (6GB)
# CPU : Intel i5 13th gen HX series
# llama model : llama3.2:3b
# embedding : qwen3-embedding:0.6B
# Vector Search : FAISS


# ============================================================
# 1. IMPORTS
# ============================================================

# To run the Ollama model
from langchain_ollama import ChatOllama

# Converts AIMessage into a normal Python string
from langchain_core.output_parsers import StrOutputParser

# To load PDF documents
from pypdf import PdfReader

# Making prompt template for the output
from langchain_core.prompts import PromptTemplate

# This is for embedding the documents
from langchain_ollama import OllamaEmbeddings

# To covert the pages to langchain document
from langchain_core.documents import Document

# To split text
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# For Vector Embedding
import faiss
import numpy as np

# ============================================================
# 1. MODEL
# ============================================================

MODEL = 'llama3.2:3b'

# TODO: Here can put the logic to choose model based on request
# need to work on that

# Object creation that knows how to communicate to ollama
model = ChatOllama(model=MODEL, temperature=0)

# ============================================================
# 2. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()

# ============================================================
# 3. PROMPT TEMPLATE
# ============================================================
# {context} and {Question are place holder} for the template
template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template=template)


# ============================================================
# 4. LANGCHAIN CHAIN
# ============================================================

# Langchain ChainTest
ChainTest = prompt | model | parser

# ============================================================
# 5. TEST THE LLM
# ============================================================

query = "Does HDFC do investment Banking?"

# The lang chain is invoked, it goes to the chain(ChainTest).
content = ChainTest.invoke(
    {
        "context": "HDFC Bank does Investment Banking?",
        "question": query
    }
)
print(content)

# ============================================================
# 6. TEST CHAIN WITH CONTEXT
# ============================================================

content = ChainTest.invoke(
    {
        "context": "The name I was given was Santiago",
        "question": "What's my name?"
    }
)
print(content)

# ============================================================
# 7. LOAD PDF
# ============================================================

# Here we have split the documents have been split into pages

# Loader holds the opened pdf
loader = PdfReader("HDFC.pdf")

# Empty list to store the content of a page as a single element(string) and all the pages are stored here
pages = []
# traversing the doc page by page
# the pages are converted into separate dictionary inside the list

for page_number, page in enumerate(loader.pages):
    text = page.extract_text()

    pages.append({
        "page_content": text,
        "metadata": {
            "source": "ConCall_HDFC.pdf",
            "page": page_number
        }
    })

print("The document start #########################################################")
for page in pages:
    print(page["page_content"])

print("The document end ############################################################")

# ============================================================
# 8. PROMPT TEMPLATE TEST
# ============================================================

formatted_text = prompt.format(
    context="Here is some context",
    question="Here is some question"
)
print(formatted_text)


# ============================================================
# 9. CONVERTING THE PAGES TO LANGCHAIN DOCUMENTS
# ============================================================

# each dictionary ( containing one page and metadata) is converted into the Langchain Document
documents = [Document(
    page_content = page["page_content"],
    metadata = page["metadata"]
    )
    for page in pages
]

# the text splitter uses it to convert the passages and paras into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
print(type(text_splitter))
print(type(documents))
print(type(documents[0]))

# Stores list of smaller document
chunks = text_splitter.split_documents(documents)
print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))


print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6B")


# ============================================================
# 10. CREATE EMBEDDINGS FOR THE CHUNKS
# ============================================================

texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)
# The vectors are stored in the RAM at this point and is not saved to disk

print("Number of chunks:", len(chunks))
print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))
print("First 10 values:", vectors[0][:10])

# That is 69 rows, with 1,024 values per row.
#           1024 dimensions
#       ┌───────────────────────┐
# Row 0 │ 0.0093, -0.0327, ...  │
# Row 1 │ 0.0124,  0.0181, ...  │
# Row 2 │ -0.031,  0.0072, ...  │
#  ...  │          ...          │
# Row 68│          ...          │
#       └───────────────────────┘
#           69 vectors


# ============================================================
# 11. CREATE FAISS VECTOR INDEX
# ============================================================

# Convert embeddings to NumPy float32
vectors_np = np.array(vectors, dtype="float32")

# Get embedding dimension
dimension = vectors_np.shape[1]

# Create FAISS index
# this creates an empty FAISS index
# We are using Euclidean distance (L2 Distance) compare vectors
index = faiss.IndexFlatL2(dimension)

# This is to insert all the generated vectors into the index
index.add(vectors_np)

# FAISS vector ID	Corresponding original chunk
# 0	                chunks[0]
# 1	                chunks[1]
# 2	                chunks[2]
# ...	            ...
# 68	            chunks[68]

print("Number of vectors in FAISS:", index.ntotal)
print("Vector dimension:", index.d)

# ============================================================
# 12. SAVE FAISS INDEX
# ============================================================

# The documents embedding is saved at the location with indexing
# TODO: I need to look for alternative of FAISS.
EMBEDDING_STORE = r"E:\Stock market data\RAG\vector_store\HDFC"

os.makedirs(EMBEDDING_STORE, exist_ok=True)

faiss_index_path = os.path.join(
    EMBEDDING_STORE,
    "index.faiss"
)

faiss.write_index(index, faiss_index_path)

print("FAISS index saved to:")
print(faiss_index_path)

# ============================================================
# 13. SAVE CHUNKS AND METADATA
# ============================================================

import json

metadata_path = os.path.join(
    EMBEDDING_STORE,
    "chunks.json"
)

chunk_data = []

for chunk_id, chunk in enumerate(chunks):
    chunk_data.append({
        "chunk_id": chunk_id,
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    })

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(chunk_data, file, ensure_ascii=False, indent=2)

print("Chunks and metadata saved to:")
print(metadata_path)

# ============================================================
# 14. TEST SEMANTIC SEARCH
# ============================================================

print("========================================================")
query = "What is the role of HDFC?"

# Convert the query into a 1024-dimensional embedding
query_vector = embeddings.embed_query(query)

# Convert to NumPy array
query_vector_np = np.array(
    [query_vector],
    dtype="float32"
)

# Search for the 5 most similar chunks
k = 5

distances, indices = index.search(
    query_vector_np,
    k
)

print("Query:")
print(query)

print("\nRetrieved chunks:")
for rank, (distance, chunk_id) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):
    print("\n" + "=" * 70)
    print("Rank:", rank)
    print("Chunk ID:", chunk_id)
    print("Distance:", distance)
    print("Page:", chunks[chunk_id].metadata["page"])
    print("\nText:")
    print(chunks[chunk_id].page_content)
