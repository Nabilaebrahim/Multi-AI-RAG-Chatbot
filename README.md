# Multi-AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangGraph, LangChain, Groq LLM, HuggingFace embeddings, and DataStax Astra DB (Cassandra). The system intelligently routes user questions to either a private vector knowledge base or Wikipedia using a smart LLM-powered router — all running inside Google Colab.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack](#tech-stack)
4. [Prerequisites & API Keys](#prerequisites--api-keys)
   - [Groq API Key](#1-groq-api-key)
   - [HuggingFace Token](#2-huggingface-token)
   - [DataStax Astra DB (Cassandra)](#3-datastax-astra-db-cassandra)
   - [Adding Keys to Google Colab](#4-adding-keys-to-google-colab-secrets)
5. [Project Structure](#project-structure)
6. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Step 1 — Install Dependencies](#step-1--install-dependencies)
   - [Step 2 — Connect to Groq](#step-2--connect-to-groq)
   - [Step 3 — Connect to Astra DB](#step-3--connect-to-astra-db)
   - [Step 4 — Build the RAG Index](#step-4--build-the-rag-index)
   - [Step 5 — Create Embeddings](#step-5--create-embeddings)
   - [Step 6 — Store Documents in Astra DB](#step-6--store-documents-in-astra-db)
   - [Step 7 — Build the Router](#step-7--build-the-router)
   - [Step 8 — Define Tools](#step-8--define-tools)
   - [Step 9 — Define Graph State & Nodes](#step-9--define-graph-state--nodes)
   - [Step 10 — Build & Run the LangGraph Workflow](#step-10--build--run-the-langgraph-workflow)
7. [The Router — How Routing Works](#the-router--how-routing-works)
8. [The RAG Pipeline](#the-rag-pipeline)
9. [The Agents & Nodes](#the-agents--nodes)
10. [Graph Conditions (Edges)](#graph-conditions-edges)
11. [What We've Achieved So Far](#what-weve-achieved-so-far)
12. [What's Next](#whats-next)

---

## Project Goal

The goal of this project is to build a **smart, multi-source question-answering chatbot** that:

- Accepts a user question in natural language.
- **Automatically decides** whether to answer from a curated private knowledge base (vector store) or fetch live information from **Wikipedia**.
- Uses a **LangGraph state machine** to orchestrate the routing and retrieval workflow.
- Stores and retrieves document embeddings from **DataStax Astra DB** (a serverless, cloud-native Cassandra vector database).
- Uses **Groq's ultra-fast LLM inference** (Qwen model) as the language brain for routing decisions.
- Uses **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) for free, locally-computed embeddings.

This is a hands-on implementation of **Adaptive RAG** — a design pattern where the retrieval path is dynamically chosen based on the nature of the question, rather than always hitting one data source.

---

## Architecture Overview

```
User Question
      │
      ▼
┌─────────────┐
│   Router    │  ← Groq LLM (Qwen) decides:
│  (LangGraph │    "vectorstore" or "wiki_search"
│    Edge)    │
└──────┬──────┘
       │
  ┌────┴─────┐
  │          │
  ▼          ▼
┌──────┐  ┌──────────┐
│ RAG  │  │Wikipedia │
│Retri-│  │  Search  │
│eval  │  │  (API)   │
└──┬───┘  └────┬─────┘
   │            │
   └─────┬──────┘
         ▼
    Documents returned
    (END of current graph)

<img width="862" height="574" alt="zo" src="https://github.com/user-attachments/assets/67d45d81-b55e-4132-8aad-0698c8ed0878" />

```

The graph is built with **LangGraph** (`StateGraph`), where:
- The **START** node feeds into a conditional edge (`route_question`).
- Depending on the routing decision, the graph flows to either `retrieve` (Astra DB vector store) or `wiki_search` (Wikipedia REST API).
- Both paths converge at **END**.

---

## Tech Stack

| Component | Library / Service | Purpose |
|---|---|---|
| LLM Inference | [Groq](https://console.groq.com/) + `langchain-groq` | Fast LLM for routing & generation |
| LLM Model | `qwen/qwen3.8-27b` (via Groq) | Structured output routing |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, local sentence embeddings |
| Vector Store | DataStax Astra DB (Cassandra) | Persistent cloud vector storage |
| RAG Framework | LangChain + LangGraph | Chains, retrievers, graph workflow |
| Document Loading | `WebBaseLoader` | Scrape and load web articles |
| Text Splitting | `RecursiveCharacterTextSplitter` | Chunk documents for indexing |
| Wikipedia Tool | Wikipedia REST API (direct) | Live general knowledge search |
| Arxiv Tool | `ArxivQueryRun` | Research paper search (available) |
| Runtime | Google Colab (T4 GPU) | Notebook execution environment |

---

## Prerequisites & API Keys

You need **3 credentials** to run this project. All of them are stored securely in Google Colab's **Secrets** panel — never hardcoded in the notebook.

---

### 1. Groq API Key

Groq provides blazing-fast LLM inference on specialized LPU hardware. The model used here is `qwen/qwen3.8-27b`.

**Steps to get your Groq API key:**

1. Go to [https://console.groq.com/](https://console.groq.com/) and sign up or log in.
2. In the left sidebar, click **"API Keys"**.
3. Click **"Create API Key"**, give it a name, and copy the generated key.
4. The key looks like: `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`

> **Secret name in Colab:** `GROQ_API_KEY`


---

### 2. HuggingFace Token

The embeddings model (`all-MiniLM-L6-v2`) is downloaded from HuggingFace. While this specific model is public and may not strictly require a token, having one configured is good practice and required if you switch to gated models later.

**Steps to get your HuggingFace token:**

1. Go to [https://huggingface.co/](https://huggingface.co/) and sign up or log in.
2. Click your profile icon (top right) → **Settings**.
3. In the left menu, click **"Access Tokens"**.
4. Click **"New token"**, select **Read** access, name it, and click **Generate**.
5. Copy the token — it starts with `hf_`.

> **Secret name in Colab:** `HUGGINGFACEHUB_API_TOKEN`
>
> **Note:** In the current version of the notebook, the HuggingFace embeddings model is loaded directly without needing to explicitly pass the token (it reads from the environment). If you encounter authentication errors, set `HUGGINGFACEHUB_API_TOKEN` in your Colab secrets as well.

<img width="3198" height="1812" alt="z" src="https://github.com/user-attachments/assets/24124d9f-503e-470d-a8eb-f2e44ac4e9e1" />


---

### 3. DataStax Astra DB (Cassandra)

Astra DB is a serverless, cloud-native vector database built on Apache Cassandra. It's used here to persistently store and query document embeddings.

**Steps to set up Astra DB and get your credentials:**

1. Go to [https://astra.datastax.com/](https://astra.datastax.com/) and sign up (free tier available).
2. Click **"Create Database"**.
   - **Database type:** Serverless (Vector)
   - **Database name:** e.g., `rag_chatbot_db`
   - **Provider/Region:** Pick the closest region to you
3. Wait for the database to reach **Active** status (usually 1–2 minutes).
4. **Get the Application Token:**
   - Go to your database dashboard → **"Connect"** tab → **"Application Token"**
   - Click **"Generate Token"** with the `Database Administrator` role
   - Copy the **Token** value — it starts with `AstraCS:`
5. **Get the Database ID:**
   - Still on the database dashboard, go to the **"Overview"** tab
   - Copy the **Database ID** (a UUID like `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

   <img width="3200" height="1756" alt="zz" src="https://github.com/user-attachments/assets/3f985f33-678f-4a9b-9f37-d97eb238ce3f" />




> **Secret names in Colab:**
> - `ASTRA_DB_APPLICATION_TOKEN` ← the `AstraCS:...` token
> - `ASTRA_DB_ID` ← the UUID of your database
 <img width="3192" height="1668" alt="zzz" src="https://github.com/user-attachments/assets/9cb2f1e4-9777-46d7-9053-45132f29d05f" />

---

### 4. Adding Keys to Google Colab Secrets

Google Colab has a built-in Secrets manager so you never need to paste keys directly into cells.

**Steps:**

1. Open your notebook in [Google Colab](https://colab.research.google.com/).
2. In the left sidebar, click the **🔑 Key icon** (Secrets).
3. Click **"+ Add new secret"** for each key:

   | Name | Value |
   |---|---|
   | `GROQ_API_KEY` | Your Groq key (`gsk_...`) |
   | `ASTRA_DB_APPLICATION_TOKEN` | Your Astra token (`AstraCS:...`) |
   | `ASTRA_DB_ID` | Your Astra database UUID |
   | `HUGGINGFACEHUB_API_TOKEN` | Your HuggingFace token (`hf_...`) *(optional but recommended)* |

4. Toggle **"Notebook access"** to ON for each secret.

The notebook reads these using:
```python
from google.colab import userdata
groq_api_key = userdata.get('GROQ_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = userdata.get('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = userdata.get('ASTRA_DB_ID')
```

---

## Project Structure

```
Multi-AI RAG Chatbot/
├── Multi_AI_RAG_Chatbot.ipynb   # Main notebook (all code)
├── README.md                    # This file
└── .gitignore                   # Ignores secrets and cache files
```

All logic lives in the single notebook `Multi_AI_RAG_Chatbot.ipynb`, structured as sequential code cells.

---

## Step-by-Step Walkthrough

### Step 1 — Install Dependencies

```python
!pip install langchain langgraph cassio
!pip install groq
!pip install langchain_community
!pip install -U langchain_community tiktoken langchain-groq langchainhub chromadb langchain langgraph langchain_huggingface
!pip install -qU langchain-text-splitters
!pip install arxiv wikipedia
!pip install -qU langchain-community cassandra-driver
```

All packages are installed in the Colab environment. Key packages:
- **`langchain` / `langgraph`** — core orchestration
- **`cassio`** — Cassandra / Astra DB connector
- **`langchain-groq`** — Groq LLM integration
- **`langchain_huggingface`** — HuggingFace embeddings
- **`chromadb`** — (available as an alternative local vector store)
- **`tiktoken`** — token counting for text splitting

---

### Step 2 — Connect to Groq

```python
from google.colab import userdata
from groq import Groq

groq_api_key = userdata.get('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)

# List available models
models = client.models.list()
for model in models.data:
    print(model.id)
```

This verifies your Groq connection and prints available models. The project uses `qwen/qwen3.8-27b` for the router.

---

### Step 3 — Connect to Astra DB

```python
import cassio

ASTRA_DB_APPLICATION_TOKEN = userdata.get('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = userdata.get('ASTRA_DB_ID')

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
print("connection done")
```

`cassio.init()` establishes the global Cassandra session used by all subsequent LangChain Cassandra integrations. On success, it prints `"connection done"`.

---

### Step 4 — Build the RAG Index

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Documents to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load and split
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
```

Three blog posts from [Lilian Weng's blog](https://lilianweng.github.io/) are scraped and split into 500-token chunks. These articles cover:
- **LLM-Powered Autonomous Agents** — agent systems, planning, memory, tool use
- **Prompt Engineering** — prompting techniques and best practices
- **Adversarial Attacks on LLMs** — robustness and security of language models

This is the **private knowledge base** — the vectorstore route answers questions about these topics.

---

### Step 5 — Create Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

The `all-MiniLM-L6-v2` model is downloaded from HuggingFace (~90 MB). It converts text into 384-dimensional dense vectors. This model is:
- **Free** — no API cost
- **Fast** — small and efficient
- **Good quality** — widely used for semantic search tasks

---

### Step 6 — Store Documents in Astra DB

```python
from langchain_community.vectorstores import Cassandra

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,   # uses the cassio global session
    keyspace=None   # uses the default keyspace
)

astra_vector_store.add_documents(doc_splits)
print("Inserted %i documents." % len(doc_splits))
```

This creates a table called `qa_mini_demo` in your Astra DB and inserts **88 document chunks** with their vector embeddings. The data persists in the cloud, so you don't need to re-insert on subsequent runs (unless the table is dropped).

<img width="3200" height="1802" alt="zf" src="https://github.com/user-attachments/assets/dec510af-1c51-4a18-820a-951b7bb0d480" />



A retriever is then created:
```python
retriever = astra_vector_store.as_retriever()
```

---

### Step 7 — Build the Router

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from typing import Literal

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen/qwen3.8-27b", max_tokens=50)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

question_router = route_prompt | structured_llm_router
```

The router uses **structured output** — the LLM is forced to return a Pydantic model with exactly one field (`datasource`) whose value is either `"vectorstore"` or `"wiki_search"`. This guarantees parseable, deterministic routing decisions.

---

### Step 8 — Define Tools

Two retrieval tools are available:

**Arxiv Tool** (available, not yet wired into the graph):
```python
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
```

**Wikipedia Tool** (active in the graph):
```python
def search_wikipedia_safely(query):
    """Direct Wikipedia REST API call — avoids LangChain wrapper issues."""
    url = "https://en.wikipedia.org/w/api.php"
    # ... search for article title, then fetch its extract
    return page_content

def wiki_search(state):
    question = state["question"]
    doc_content = search_wikipedia_safely(question)
    wiki_results = [Document(page_content=doc_content)]
    return {"documents": wiki_results, "question": question}
```

> **Note:** The original `WikipediaQueryRun` wrapper was replaced with a direct Wikipedia REST API call (`search_wikipedia_safely`) to avoid encoding and timeout issues that occurred in the Colab environment.

---

### Step 9 — Define Graph State & Nodes

```python
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
```

The `GraphState` is the shared data structure passed between all nodes in the LangGraph graph. Currently it carries:
- `question` — the user's input question
- `generation` — reserved for the final LLM-generated answer (next step)
- `documents` — the retrieved documents from either source

The **retrieve node** pulls from Astra DB:
```python
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```

---

### Step 10 — Build & Run the LangGraph Workflow

```python
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)

# Conditional routing from START
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)

# Both nodes end the graph
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

app = workflow.compile()
```

Run the graph:
```python
inputs = {"question": "What is agent?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
pprint(value['documents'][0].dict()['metadata']['description'])
```

<img width="862" height="574" alt="zo" src="https://github.com/user-attachments/assets/84df2e0e-2898-41c3-bece-4e5a92bbf810" />


---

## The Router — How Routing Works

The router is the decision-making brain of the chatbot. It answers the question: **"Where should I look for the answer?"**

| Question | Routes To | Reason |
|---|---|---|
| "What are the types of agent memory?" | `vectorstore` | Topic is in the indexed knowledge base |
| "Who is Taha Hessen?" | `wiki_search` | Not in the knowledge base → Wikipedia |
| "What is prompt engineering?" | `vectorstore` | Explicitly covered in indexed docs |
| "Avengers" | `wiki_search` | Unrelated to the knowledge base |

**How it works technically:**
1. The question is passed through a `ChatPromptTemplate` with a system message that describes what's in the vector store.
2. The LLM (`qwen/qwen3.8-27b` via Groq) outputs a structured `RouteQuery` Pydantic object.
3. The `datasource` field is either `"vectorstore"` or `"wiki_search"`.
4. The LangGraph conditional edge reads this value and routes the graph accordingly.

The router uses `max_tokens=50` since it only needs to output a short JSON object — this keeps it extremely fast and cheap.

---

## The RAG Pipeline

RAG (Retrieval-Augmented Generation) works in this pipeline as follows:

```
1. Documents loaded from 3 URLs (Lilian Weng blog posts)
        ↓
2. Split into 500-token chunks (88 total chunks)
        ↓
3. Embedded using HuggingFace all-MiniLM-L6-v2 (384-dim vectors)
        ↓
4. Stored in DataStax Astra DB (table: qa_mini_demo)
        ↓
5. At query time: question is embedded → top-k similar chunks retrieved
        ↓
6. Retrieved chunks returned as context (generation step coming next)
```

The knowledge base covers three topics:
- **Agents** — LLM-powered autonomous agent architectures, planning, memory, tool use
- **Prompt Engineering** — techniques like chain-of-thought, few-shot, instruction tuning
- **Adversarial Attacks on LLMs** — attack methods and robustness considerations

---

## The Agents & Nodes

In LangGraph, each **node** is a function that receives the graph state and returns an updated state. These are the equivalent of "agents" in the workflow:

| Node | Function | Description |
|---|---|---|
| `retrieve` | `retrieve(state)` | Queries Astra DB vector store using the user question, returns top-k relevant documents |
| `wiki_search` | `wiki_search(state)` | Calls the Wikipedia REST API directly, returns article extract as a Document |

The **edge function** (`route_question`) is not a node but a routing agent — it invokes the Groq LLM and returns the name of the next node to visit.

---

## Graph Conditions (Edges)

LangGraph distinguishes between **regular edges** (always go to the next node) and **conditional edges** (route based on a function's return value).

```
START
  │
  └─ [conditional_edge: route_question()]
       │
       ├── returns "wiki_search"  →  wiki_search node  →  END
       │
       └── returns "vectorstore"  →  retrieve node     →  END
```

**`route_question(state)`** is the conditional edge function:
```python
def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"
```

<img width="3192" height="1774" alt="xx" src="https://github.com/user-attachments/assets/a1f85bfe-a2c2-4087-9d82-0bb8da95f870" />
<img width="3200" height="1752" alt="xxx" src="https://github.com/user-attachments/assets/0c6822f3-cd12-4475-9e80-f16f2db58904" />



It returns a string key that LangGraph maps to the corresponding node name (defined in the `add_conditional_edges` call).

---

## What We've Achieved So Far

- [x] **Environment setup** — All dependencies installed in Colab
- [x] **Groq connection** — LLM inference via Groq API verified and working
- [x] **Astra DB connection** — Cloud vector database connected via `cassio`
- [x] **Document ingestion** — 3 web articles loaded, split into 88 chunks
- [x] **Embeddings** — HuggingFace `all-MiniLM-L6-v2` model integrated
- [x] **Vector store** — 88 documents stored in Astra DB (`qa_mini_demo` table)
- [x] **Retriever** — Vector similarity search working and tested
- [x] **LLM Router** — Groq + structured output routing questions to correct data source
- [x] **Wikipedia tool** — Direct Wikipedia REST API search implemented and working
- [x] **Arxiv tool** — Tool configured (available for future use)
- [x] **LangGraph workflow** — Full `StateGraph` compiled with conditional routing
- [x] **End-to-end test** — Routing verified: `"What is agent?"` → RAG, `"Avengers"` → Wikipedia

---

## What's Next

The current graph retrieves documents but does **not yet generate a final answer**. The planned next steps are:

- [ ] **Add generation node** — Use the retrieved documents as context and pass them to the LLM to generate a final natural-language answer
- [ ] **Add grading node** — Grade retrieved documents for relevance before generation (document grader)
- [ ] **Add hallucination grader** — Check whether the generation is grounded in the retrieved documents
- [ ] **Add answer grader** — Check whether the generation actually addresses the question
- [ ] **Add re-routing on failure** — If documents are irrelevant, route back to Wikipedia or trigger a re-write of the question
- [ ] **Add Arxiv routing** — Wire the Arxiv tool into the router as a third data source for research paper questions
- [ ] **Add conversational memory** — Maintain chat history across turns
- [ ] **Build a UI** — Add a Gradio or Streamlit interface for interactive use

---

## Notes & Troubleshooting

**Wikipedia LangChain wrapper issues:**
The standard `WikipediaQueryRun` tool sometimes fails in Colab due to encoding or dependency issues. This project replaces it with a direct Wikipedia REST API call using `requests`, which is more reliable.

**Astra DB consistency warning:**
You may see this warning during retrieval:
```
Server warning: Top-K queries can only be run with consistency level ONE / LOCAL_ONE / NODE_LOCAL.
Consistency level LOCAL_QUORUM was requested. Downgrading to LOCAL_ONE.
```
This is expected behavior — Astra DB automatically adjusts the consistency level for vector similarity (ANN) queries. It does not affect correctness.

**Re-inserting documents:**
The `astra_vector_store.add_documents(doc_splits)` cell will add documents again if run multiple times. If you want to avoid duplicates on re-runs, skip this cell after the first successful run (the data persists in Astra DB).

**Model availability on Groq:**
The available models change over time. Run the model listing cell at the start to confirm `qwen/qwen3.8-27b` is available. If not, update the `model_name` parameter in the router cell to any other available chat model.
