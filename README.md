# LangChain Chatbot (chain_intro)

Small example project demonstrating a LangChain-based chatbot with:
- OpenAI LLM (via langchain_openai)
- Chroma vector store for retrieval
- A simple Gradio UI to interact with an agent composed of chains/tools

## Description

This retrieval-augmented chatbot combines a LangChain agent, OpenAI chat models, and a local Chroma vector store to answer domain-specific questions. It routes queries to domain tools (Review, Operating_System, Waits (function call)), retrieves supporting documents, and produces grounded responses via prompt templates.
Features
- Retrieval-augmented generation (RAG) using Chroma + OpenAI embeddings
- Domain-specific chains (reviews, OS doc, personal/career)
- Tool-based agent orchestration with function-calling style agent
- Gradio UI for local testing and sharing
- Verbose agent mode to inspect intermediate steps for debugging

How to populate Chroma (brief)
- Collect documents (text / PDFs)
- Create embeddings with OpenAIEmbeddings and persist via Chroma client
- Ensure `chroma_data/` contains persisted collections before running chatbot

Extending the bot
- Add new domain/chains: create a new ChatPromptTemplate, wire a chain with the retriever, and expose it as a Tool.
- Swap embedding model or vector DB: replace OpenAIEmbeddings or Chroma with other supported implementations.
- Add authentication & rate limiting in production (wrap agent calls, queue requests, or proxy through a backend).
  
Architecture overview

Retriever-augmented pipelines: The app builds separate chains for different contexts (reviews_chain, os_chain, function call). Each chain:
  - Accepts {"context": retriever, "question": RunnablePassthrough()}
  - Applies a ChatPromptTemplate to combine retrieved context with the user question
  - Invokes ChatOpenAI to produce the final answer
  - Parses output with StrOutputParser
Vector store: Chroma persists embeddings at `chroma_data/`. OpenAI embeddings (OpenAIEmbeddings) are used to index documents and power the retriever (`as_retriever(k=10)`).

Agent & Tools: create_openai_functions_agent + AgentExecutor wrap the chains as Tools:
  - Reviews -> reviews_chain.invoke (answers about patient reviews)
  - Operating_System -> os_chain.invoke (answers strictly from an "OPERATING SYSTEM" document)
  - Waits -> get_current_wait_time (returns numeric wait times)
  The agent can call tools programmatically and returns intermediate steps when executed.
UI: Gradio interface (`gr.Interface`) exposes ChatPot which sends user input into the agent executor and returns text responses.

Prompt templates and safety
- System prompts restrict the LLM to only use context from the retriever and to answer "I don't know." when the context is insufficient.
- Multiple templates exist to constrain responses per domain (reviews, OS document, personal/career info).
- Keep system/human prompt templates concise and explicit to minimize hallucination.

Data flow (textual)
1. User message -> Gradio -> ChatPot
2. Agent parses intent -> chooses a tool
3. If a retrieval tool is chosen: retriever fetches top-k docs -> prompt template formats context + question -> LLM -> StrOutputParser -> tool returns a clean string
4. Agent assembles final output and Gradio displays it

Troubleshooting & common issues
- 401 / invalid_api_key:
  - Ensure OPENAI_API_KEY environment variable is set to a valid user key from https://platform.openai.com/account/api-keys.
  - Managed/project keys (e.g., `sk-proj-...`) may be rejected; if you see "Incorrect API key provided" replace with a user key.
  - Confirm the environment running the script (IDE, terminal, container) has the variable exported.
- Empty retrieval results:
  - Verify Chroma contains documents and embeddings. Use a small test script to embed a sample doc and confirm retriever returns it.
- Dependency mismatches:
  - Use a virtualenv and pin versions. If you need a requirements.txt, request it and I'll generate one.

Files of interest
- chain_intro/chatbot.py — main script (chains, tools, agent, Gradio UI)
- tools.py — helper functions (e.g., get_current_wait_time)
- chroma_data/ — persistent data folder used by the Chroma vector store

Notes
- Keep API keys secret. Do not commit .env files containing secrets.
- For production, move Gradio behind authentication and consider running the agent in an async worker process.


Quick start

1. Clone / copy the project into a working directory.

2. Create and activate a Python virtual environment
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows (cmd/powershell)

3. Install dependencies
   pip install -r requirements.txt
   (If requirements.txt is not present, install manually:
    pip install langchain langchain-openai langchain-community chromadb gradio python-dotenv)

4. Configure environment variables
   - Create a .env file at the project root or export the variable in your shell:
     OPENAI_API_KEY=sk-...
   - The chatbot expects OPENAI_API_KEY to be set. If you use a managed/project key (e.g., starting with `sk-proj-`) you may receive authentication errors from OpenAI; replace it with a full user key if needed.

5. Prepare vector store data
   - The code expects a Chroma directory at `chroma_data/`. Populate or persist embeddings there, or run example embedding scripts to create it before launching.
6. Run the chatbot
   python chatbot.py
   - This will start a Gradio UI by default and print a local URL.




