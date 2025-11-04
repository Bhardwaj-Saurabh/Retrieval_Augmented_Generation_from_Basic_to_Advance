# 🚀 Retrieval-Augmented Generation (RAG)
### From Basics to Production-Ready Systems


<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)](RAG_notebooks)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-Powered-green)](https://www.langchain.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

*A hands-on course to master RAG systems through practical implementations and real-world applications*

[📚 Getting Started](#-getting-started) • [📖 Course Content](#-course-curriculum) • [💻 Tech Stack](#-tech-stack) • [🎯 Who's This For](#-whos-this-for)

</div>

![RAG](images/wallpaper.png)

---

## 🌟 What is RAG?

**Retrieval-Augmented Generation (RAG)** is the breakthrough technique powering modern AI applications like ChatGPT plugins, Notion AI, and enterprise document assistants. RAG combines:

- 🔍 **Retrieval** from external knowledge stores
- 🤖 **Generation** using Large Language Models (LLMs)
- ✨ **Context-aware** responses that are factual and up-to-date

Instead of relying solely on an LLM's training data, RAG systems fetch relevant information in real-time and use it to generate accurate, grounded responses.

---

## 🎯 What You'll Build

This comprehensive course takes you from **zero to production** in 8 weeks. You'll master:

✅ **Pythonic RAG** — Build from scratch to understand core principles  
✅ **Vanilla RAG** — Implement standard RAG patterns  
✅ **Advanced RAG** — Optimize retrieval, chunking, and generation  
✅ **LangChain Integration** — Leverage industry-standard frameworks  
✅ **Vector Databases** — Work with FAISS and ChromaDB  
✅ **Production Deployment** — Deploy real-world RAG applications  

---

## 📖 Course Curriculum

**Duration:** 8 weeks (2 lessons per week)  
**Level:** Intermediate  
**Format:** Blog posts + Interactive notebooks + Production code

| Week | Lesson | Title | Blog | Notebook |
|------|--------|-------|------|----------|
| **Week 1** | 1.1 | Building Pythonic RAG from Scratch | [📖 Read](RAG_notebooks/1.%20Pythonic_RAG/Basic_RAG.md) | [📓 Open](RAG_notebooks/1.%20Pythonic_RAG/Basic_RAG.ipynb) |
| **Week 2** | 2.1 | Implementing Vanilla RAG | [📖 Read](RAG_notebooks/2.%20Vanilla_RAG/vanilla_rag.md) | [📓 Open](RAG_notebooks/2.%20Vanilla_RAG/vanilla_rag.ipynb) |
| **Week 3** | 3.1 | Building Blocks of RAG | [📖 Read](RAG_notebooks/3.%20Deep_Dive_RAG/building_blocks_of_RAG.md) | [📓 Open](RAG_notebooks/3.%20Deep_Dive_RAG/building_blocks_of_RAG.ipynb) |
| **Week 3** | 3.2 | RAG with LangChain | [📖 Read](RAG_notebooks/3.%20Deep_Dive_RAG/building_blocks_of_RAG.md) | [📓 Open](RAG_notebooks/3.%20Deep_Dive_RAG/Simple_RAG_System_Langchain.ipynb) |

> 📌 **More lessons coming soon!** This is an active course with new content added regularly.

---

## 🚀 Getting Started

### Prerequisites

Before diving in, make sure you have:

- **Python 3.11+** installed
- Basic understanding of:
  - Python programming
  - Machine Learning fundamentals
  - Large Language Models (LLMs)
  - Vector embeddings (helpful but not required)

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rag-course.git
cd rag-course

# 2. Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
uv sync
```

### Launch Jupyter

```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or use Jupyter Notebook
jupyter notebook
```

Navigate to `RAG_notebooks/` and open any lesson to start learning!

---

## 📁 Repository Structure

```
rag-course/
│
├── 📓 RAG_notebooks/           # Interactive lesson notebooks
│   ├── 1. Pythonic_RAG/        # Week 1: Build from scratch
│   ├── 2. Vanilla_RAG/         # Week 2: Standard RAG patterns
│   └── 3. Deep_Dive_RAG/       # Week 3: Advanced techniques
│
├── 🖼️ images/                  # Diagrams and visualizations
├── 📄 rag_docs/                # Data & resources
├── 🐍 pyproject.toml           # Project metadata
├── 📋 requirements.txt         # Python dependencies
└── 📖 README.md                # You are here!
```

---

## 💡 How to Use This Course

Follow this proven learning path:

1. **📖 Read the Blog** — Understand concepts and theory
2. **📓 Run the Notebook** — See implementations in action
3. **💻 Experiment** — Modify code and test your ideas
4. **🔗 Check References** — Dive deeper with additional resources
5. **🛠️ Build Your Own** — Apply what you learned to real projects

---

## 🛠️ Tech Stack

This course uses industry-standard tools:

| Category | Technologies |
|----------|-------------|
| **LLMs** | OpenAI GPT-4, Anthropic Claude, Local models |
| **Vector DBs** | FAISS, ChromaDB |
| **Embeddings** | OpenAI Embeddings, Sentence Transformers |
| **Frameworks** | LangChain, LlamaIndex |
| **Deployment** | FastAPI, Docker, Cloud platforms |
| **Evaluation** | RAGAS, Custom metrics |

---

## 🎓 Who's This For?

This course is perfect for:

👨‍💻 **Developers** building AI-powered applications  
📊 **Data Scientists** implementing production ML systems  
🏢 **ML Engineers** deploying LLM solutions  
🎓 **Students** learning modern AI architectures  
🚀 **Founders** prototyping AI products  

**You'll get the most value if you:**
- Know Python basics
- Understand ML fundamentals
- Want hands-on, practical experience (not just theory)

---

## 🌟 What Makes This Course Different?

✨ **No Fluff** — Pure hands-on implementations, no endless theory  
🔧 **Production-Ready** — Learn patterns used by real companies  
📈 **Progressive Learning** — From simple to advanced, step by step  
🎯 **Goal-Oriented** — Build actual working systems you can deploy  
🆓 **Open Source** — All code is free and available forever  

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- 🐛 **Report bugs** — Open an issue if something doesn't work
- 📝 **Improve docs** — Fix typos, clarify explanations
- 💻 **Add examples** — Contribute new use cases or implementations
- 🌟 **Share feedback** — Tell us what worked and what didn't

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Additional Resources

- 🔗 **Official Docs** — [LangChain](https://docs.langchain.com) | [OpenAI](https://platform.openai.com/docs)
- 💬 **Community** — Join discussions in Issues

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to:
- The open-source community for amazing tools
- Researchers advancing RAG and LLM technology
- Everyone contributing to this course

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

**Questions?** Open an issue or reach out!

Made with ❤️ by developers, for developers

[⬆ Back to Top](#-retrieval-augmented-generation-rag)

</div>