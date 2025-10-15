# 1. What is Retrieval-Augmented Generation (RAG)?

**Retrieval-Augmented Generation (RAG)** is a technique that improves how large language models (LLMs) — such as ChatGPT — answer questions more accurately and contextually.

LLMs are powerful tools capable of answering questions, summarizing text, generating code, and much more. However, they rely only on the data they were trained on, meaning they might not know about **recent events** or **specialized topics**.

![LLM](../../images/llm_interaction.png)
*Source: RAG Coursera Course*

RAG addresses this limitation by adding a **retrieval step** before the model generates its response.

### 1.1 How RAG Works

**1. Retrieval:**  
   When a user asks a question, the system first searches a **knowledge base** (a collection of reliable, possibly private, or domain-specific documents) to find the most relevant information related to the query.

**2. Augmentation:**  
   The retrieved information is then **added to the user’s original prompt**, creating an **augmented prompt** that provides additional context.

**3. Generation:**  
   The LLM uses this augmented prompt to **generate a more accurate, up-to-date, and context-aware response**.

![LLM](../../images/vanilla_rag.svg)

### 1.2 Why RAG Matters

This approach helps LLMs perform better on tasks that require **specific**, **current**, or **specialized information** — just like how humans first gather information before reasoning and responding.  

---

## 2. Applications of RAG (Retrieval-Augmented Generation)

**Retrieval-Augmented Generation (RAG)** enhances large language models (LLMs) by combining them with an external knowledge base containing information the model didn’t see during training. This helps LLMs generate more accurate, context-aware, and up-to-date responses.  

Below are the main **applications of RAG**:

### 2.1. Code Generation
- While LLMs are trained on large amounts of public code, they may not know project-specific details such as custom **classes, functions, or coding styles**.  
- A RAG system can use a **project’s codebase as the knowledge base**, allowing the model to retrieve relevant files and definitions before generating or explaining code.  
- This leads to **more accurate and context-relevant code outputs**.

### 2.2. Enterprise Chatbots
- Each company has its own **products, policies, and documentation**.  
- By treating these as a **knowledge base**, a RAG-powered chatbot can:
  - Answer customer questions about products, pricing, or inventory.  
  - Help employees find internal policy information or documentation.  
- This ensures **accurate, company-specific responses** and reduces the risk of **hallucinations** or generic replies.

### 2.3. Healthcare and Legal Domains
- These fields demand **precision** and often rely on **private, specialized data** (e.g., medical journals, case files, or legal documents).  
- RAG allows LLMs to access and use this domain-specific information safely and accurately.  
- This approach ensures **high-quality, factual, and compliant responses**.

### 2.4. AI-Assisted Web Search
- Modern search engines now integrate RAG-like systems.  
- They **retrieve information from the web** and then **generate AI summaries** that present key points quickly.  
- Essentially, these systems use the **entire internet as the knowledge base**, providing efficient and intelligent search summaries.

### 2.5. Personalized AI Assistants
- Smaller RAG systems can enhance personal productivity tools such as **email clients, calendars, and text editors**.  
- The knowledge base might include:
  - Personal emails  
  - Contact lists  
  - Documents or notes  
- This allows assistants to generate responses, drafts, or reminders that are **highly personalized and context-aware**.