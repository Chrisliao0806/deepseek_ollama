# deepseek_ollama

## Overview
`deepseek_ollama` is a project that utilizes the DeepSeek-R1 model from [Ollama](https://ollama.com/library/deepseek-r1) to perform advanced data analysis.

## Installation

### Prerequisites
- Python 3.10 or higher
- asyncio
- ollama

### Install Ollama

To install the Ollama package, go to [https://ollama.com/](https://ollama.com/), click the "Download" button, and select your operating system for different installation methods. Then, run the following command:
```bash
pip install ollama
```

## Usage
To run the program, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Chrisliao0806/deepseek_ollama.git
    cd deepseek_ollama
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running `easy_response.py`
This script allows you to ask a question to the DeepSeek-R1 model.

Example:
```bash
python easy_response.py --question "Your question here"
```

Using `--remove-think` Option:
The `--remove-think` option allows you to remove content between `<think>` and `</think>` tags from the model's response.

Example without `--remove-think`:
```bash
python easy_response.py --question "Your question here"
```

Example with `--remove-think`:
```bash
python easy_response.py --question "Your question here" --remove-think True
```

### Running `milvus_rag.py`
This script performs Retrieval-Augmented Generation (RAG) using Milvus for vector storage and retrieval.

Example:
```bash
python milvus_rag.py --pdf-file "path/to/your.pdf" --question "Your question here"
```

### Running `rag.py`
This script reads PDF files and creates a RetrievalQA chain.

Example:
```bash
python rag.py --pdf-file "path/to/your.pdf" --question "Your question here"
```

### Running `adapative_rag.py`
This script implements a retrieval-augmented generation (RAG) system for processing and answering questions based on a given PDF document.

Example:
```bash
python adapative_rag.py --pdf-file "path/to/your.pdf" --question "Your question here"
```

## Model
This project uses the DeepSeek-R1 model from [Ollama](https://ollama.com/library/deepseek-r1). Ensure you have the model downloaded and properly configured in your environment.

## License
This project is licensed under the MIT License.
