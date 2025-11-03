# RAG Fundamentals: Understanding the Building Blocks

Before diving into building Retrieval-Augmented Generation (RAG) systems, we need to understand the fundamental concepts that make RAG possible. This guide will walk you through the essential building blocks: tokens, tokenization, embeddings, vector search, and vector databases.

## Table of Contents

1. [What is a Token?](#what-is-a-token)
2. [Understanding Tokenization](#understanding-tokenization)
3. [What are Embeddings?](#what-are-embeddings)
4. [How Are Embeddings Trained?](#how-are-embeddings-trained)
5. [How Does Vector Search Work?](#how-does-vector-search-work)
6. [What is a Vector Database?](#what-is-a-vector-database)
7. [Traditional Database vs Vector Database](#traditional-database-vs-vector-database)
8. [Putting It All Together](#putting-it-all-together-rag-pipeline)

## What is a Token?

A **token** is the smallest unit of text that a language model can process. Think of tokens as the "words" that AI models understand.

### How Tokenization Works

When you input text into an AI model, it doesn't process full words or sentences directly. Instead, it breaks the text into tokens.

**Rule of thumb**: 1 token ≈ 0.75 words (or ~4 characters in English)

## Understanding Tokenization

**Tokenization** is the process of breaking text into tokens. It's the critical first step before any AI processing happens.

### Types of Tokenization

#### 1. **Word-Based Tokenization**
- Splits text by spaces and punctuation

```
Input: "unhappiness is bad"
Tokens: ["unhappiness", "is", "bad"]
```

#### 2. **Character-Based Tokenization**
- Each character is a token

```
Input: "ChatGPT"  
Tokens: ["C", "h", "a", "t", "G", "P", T"]
```

#### 3. **Subword Tokenization** (Modern Standard)
- Breaks words into meaningful subunits

```
Input: "unhappiness is bad"
Tokens: ["un", "happiness", "is", "bad"]
```

## From Tokens to Token IDs
Because LLM Models can only process numerical inputs, not text strings, after text is split into tokens, each token is converted to a token ID - a unique integer that represents that token in the model's vocabulary.

```
Text: "Hello world"
↓
Tokens: ["Hello", " world"]
↓
Token IDs: [15496, 1917]
```

### Different Models, Different Tokenizers

Important: The same text can have different token counts across different models.

```
Text: "Artificial Intelligence is transforming industries"

GPT-4 tokenizer: 7 tokens
Claude tokenizer: 8 tokens  
Llama tokenizer: 9 tokens
```

## What are Embeddings?

Token IDs represents text as numbers, but they fail to capture the meaning of words and relationship of words within a corpus. This is where **embedding** plays an important role. **Embeddings** are numerical representations of text that capture semantic meaning. They convert words, sentences, or documents into vectors (arrays of numbers) that machines can understand and compare.

### The Core Idea

Similar meanings → Similar vectors

```
"king" → [0.8, 0.9, 0.1, ...]
"queen" → [0.7, 0.85, 0.15, ...]
"car" → [0.1, 0.2, 0.8, ...]
```

In this simplified example, "king" and "queen" have similar vectors because they have related meanings, while "car" is very different.

```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```

This shows embeddings understand that:
- King is to man as queen is to woman
- The "royalty" concept is preserved
- The "gender" concept is transformed

### Types of Embeddings for RAG

**1. Word Embeddings**
- One vector per word

**2. Sentence Embeddings**
- One vector per sentence

**3. Document Embeddings**
- One vector per document/paragraph

**4. Token Embeddings**
- One vector per token

## How Are Embeddings Trained?

Embeddings are created using neural networks trained on massive amounts of text data. Here's how the training process works.

### The Training Process
1. **Collect massive text data**: Wikipedia, books, websites, scientific papers
2. **Define training objective**: E.g., predict masked words
3. **Train neural network**: Millions to billions of parameters
4. **Learn patterns**: Which words/phrases appear together
5. **Create vector space**: Where similar meanings cluster together

## How Does Vector Search Work?

Vector search (also called semantic search or similarity search) finds similar items by comparing their embeddings in vector space.

### The Basic Process

**Step 1: Convert query to embedding**
```
User query: "How does machine learning work?"
→ Embedding: [0.23, 0.67, 0.12, ..., 0.89]
```

**Step 2: Compare with stored embeddings**
Calculate similarity between query embedding and all document embeddings in the database.

**Step 3: Rank by similarity**
Sort documents by similarity score (highest to lowest).

**Step 4: Return top results**
Return the k most similar documents (e.g., top 5).

### Measuring Similarity

There are several mathematical ways to measure how similar two vectors are:

#### 1. **Cosine Similarity** (Most Common in RAG)

Measures the angle between two vectors, ignoring magnitude.

**Range**: -1 to 1 (1 = identical direction, 0 = perpendicular, -1 = opposite)

**Formula**: similarity = (A · B) / (||A|| × ||B||)

**Why it's used**:
- Direction matters more than magnitude for semantic meaning
- Normalized (always between -1 and 1)
- Fast to compute
- Works well for high-dimensional spaces

**Example interpretation**:
- 0.9-1.0: Very similar
- 0.7-0.9: Similar
- 0.5-0.7: Somewhat related
- Below 0.5: Not very related

#### 2. **Euclidean Distance** (L2 Distance)

Straight-line distance between two points in space.

**Range**: 0 to ∞ (0 = identical, larger = more different)

**Formula**: distance = √(Σ(A_i - B_i)²)

**When to use**:
- When magnitude matters
- For image embeddings
- When embeddings are already normalized

#### 3. **Dot Product**

Simple multiplication of corresponding elements.

**Range**: -∞ to ∞ (higher = more similar)

**Formula**: similarity = Σ(A_i × B_i)

**When to use**:
- When embeddings are normalized (equivalent to cosine similarity)
- Fastest to compute
- Good for approximate search

#### 4. **Manhattan Distance** (L1 Distance)

Sum of absolute differences.

**Range**: 0 to ∞ (0 = identical)

**Formula**: distance = Σ|A_i - B_i|

**When to use**:
- More robust to outliers than Euclidean
- Less common in RAG

### Vector Search in Action

**Scenario**: You have 1,000 documents embedded in your vector database.

**Query**: "What is artificial intelligence?"



## What is a Vector Database?

A **vector database** is a specialized database designed to store, index, and efficiently search high-dimensional vectors (embeddings).

### Why Do We Need Vector Databases?

**The Problem**: Traditional databases can't efficiently handle vector similarity search.

Naive approach for 1 million documents:

```
For each document in database:
    Calculate similarity(query_vector, document_vector)
Sort by similarity
Return top k results
```

This is **O(n) complexity** - you must check EVERY single vector!

- 1 million vectors × 768 dimensions = ~750 million calculations
- Unacceptably slow for real-time applications

**The Solution**: Vector databases use specialized data structures and algorithms to make search sub-linear (O(log n) or even O(1)).


## Traditional Database vs Vector Database

Understanding the fundamental differences helps you architect better RAG systems.

### Data Storage Comparison

#### **Traditional Database (SQL/NoSQL)**

**Structure**:
```
Table: documents
+----+-------------------+---------------------------+
| id | title             | content                   |
+----+-------------------+---------------------------+
| 1  | "Intro to ML"     | "Machine learning is..."  |
| 2  | "Python Guide"    | "Python is a language..." |
+----+-------------------+---------------------------+
```

**Storage**: Rows and columns of structured data

**Search**: SQL queries with WHERE clauses

#### **Vector Database**

**Structure**:
```
Collection: documents
+----+---------------------------+------------------------+
| id | embedding                 | metadata               |
+----+---------------------------+------------------------+
| 1  | [0.1, 0.2, ..., 0.8]     | {title: "Intro to ML"} |
| 2  | [0.3, 0.1, ..., 0.6]     | {title: "Python Guide"}|
+----+---------------------------+------------------------+
```

**Storage**: High-dimensional vectors with optional metadata

**Search**: Similarity calculations in vector space

### Search Comparison

#### **Traditional Database Search**

**Exact Match**:
```
SELECT * FROM documents 
WHERE content LIKE '%machine learning%'
```

**Result**: Documents containing exact phrase "machine learning"

**Characteristics**:
- Finds exact text matches
- Case-sensitive (unless specified)
- Misses synonyms and related concepts
- Fast for indexed columns
- Boolean logic (AND, OR, NOT)

#### **Vector Database Search**

**Semantic Search**:
```
Query: "machine learning"
Query embedding: [0.12, 0.25, ..., 0.83]

Find: Most similar embeddings in vector space
```

