# Architecture of RAG

## 1. Overview

A **RAG system** enhances the capabilities of a **Large Language Model (LLM)** by connecting it to an external **knowledge base** through a **retriever** component. This allows the LLM to access up-to-date or domain-specific information that was not part of its training data.

A RAG (Retrieval-Augmented Generation) system uses a **retriever** to fetch relevant documents from a knowledge base before passing them to an LLM. The retriever's job is to quickly identify and return the most relevant documents for a given prompt.

![LLM](../../images/vanilla_rag.svg)

### 2. Core Components RAG

### 2.1 Key Components

1. **LLM (Large Language Model):**  
   Generates the final response based on the user’s query and any additional retrieved information.

2. **Knowledge Base:**  
   A database of trusted and relevant documents — such as company policies, research papers, or other useful text — that the system can search.

3. **Retriever:**  
   The component that searches the knowledge base to find the most relevant documents related to the user’s query.

### 3. How RAG Works (Step-by-Step)

1. **User Input:**  
   The user types a prompt (e.g., *"Why are hotels in Vancouver expensive this weekend?"*).

2. **Retrieval Phase:**  
   The **retriever** queries the **knowledge base** and fetches relevant documents.

3. **Augmentation Phase:**  
   The system combines (or *augments*) the user’s original prompt with the retrieved information, forming an **augmented prompt**.  
   Example:  
   > “Answer the following question: Why are hotels in Vancouver so expensive this weekend?  
   > Here are five relevant articles that may help you respond…”

4. **Generation Phase:**  
   The augmented prompt is sent to the **LLM**, which generates a response using both its internal knowledge and the retrieved context.

5. **Response:**  
   The user receives an accurate, up-to-date, and context-aware answer — with a similar user experience to interacting with a normal LLM, though with slightly more latency.

## 4. Advantages of RAG

1. **Access to External Information:**  
   Enables LLMs to use information not available during their training (e.g., company data, private documents, recent news).

2. **Reduces Hallucinations:**  
   Grounding the LLM’s responses in retrieved, factual data reduces the chance of generating false or generic content.

3. **Keeps Models Up to Date:**  
   Updating the **knowledge base** (instead of retraining the model) ensures that the system always has access to current information.

4. **Supports Source Citation:**  
   RAG systems can include source citations in the augmented prompt, allowing the LLM to reference them in its answers and improve transparency.

5. **Improved Efficiency:**  
   The **retriever** focuses on finding relevant facts, while the **LLM** focuses on generating natural, coherent text — allowing each to perform its strongest role.
