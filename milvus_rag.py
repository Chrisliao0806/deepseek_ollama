import re
import asyncio
import argparse
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from utils.logger import setup_logging

PROMPT_TEMPLATE = """
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:"""


def parse_arguments():
    """
    Parses command-line arguments for the RAG model using langchain.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Arguments:
        --pdf-file (str): The path to the PDF file. Default is "Chris_Resume.pdf".
        --model-path (str): The path to the DeepSeek-R1 model file. Default is "deepseek-r1:14b".
        --chunk-size (int): The maximum size of each text chunk. Default is 300.
        --chunk-overlap (int): The number of characters that overlap between chunks. Default is 10.
        --model-name (str): The name of the model to use for embedding. Default is "sentence-transformers/all-MiniLM-L6-v2".
        --question (str): The question to ask the model. Default is "What engineer is him, and what did he do".
        --db-save-dir (str): The directory to save the vector database. Default is "./milvus_example.db".
        --remove-think (bool): If True, removes content between <think> and </think> tags from the response. Default is False.
        --log-level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".
    """
    parser = argparse.ArgumentParser(description="RAG model for using langchain")
    parser.add_argument(
        "--pdf-file",
        default="Chris_Resume.pdf",
        type=str,
        help="The path to the PDF file.",
    )
    parser.add_argument(
        "--model-path",
        default="deepseek-r1:14b",
        type=str,
        help="The path to the DeepSeek-R1 model file.",
    )
    parser.add_argument(
        "--chunk-size",
        default=300,
        type=int,
        help="The maximum size of each text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        default=10,
        type=int,
        help="The number of characters that overlap between chunks.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="The name of the model to use for embedding.",
    )
    parser.add_argument(
        "--question",
        default="What engineer is him, and what did he do",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--db-save-dir",
        default="./milvus_example.db",
        type=str,
        help="The directory to save the vector database.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help="The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    return parser.parse_args()


class MilvusRag:
    """
    A class to perform Retrieval-Augmented Generation (RAG) using Milvus for vector storage and retrieval.

    Attributes:
        pdf_reader (PyMuPDFLoader): An instance of PyMuPDFLoader to read and load the PDF file.

    Methods:
        __init__(pdf_file):
            Initializes the MilvusRag instance with a PDF file.

        combine_docs(docs):
            Combines the content of multiple documents into a single string.

        rag_search(model_path="deepseek-r1:14b", query="Chris在AIF的角色是什麼？", db="./milvus_example.db", chunk_size=300, chunk_overlap=10, model_name="sentence-transformers/all-MiniLM-L6-v2", remove_think=False):
            Performs a RAG search using the specified parameters and returns the response.
    """

    def __init__(self, pdf_file):
        logging.info("Initializing RAG with PDF file: %s", pdf_file)
        self.pdf_reader = PyMuPDFLoader(pdf_file).load()

    def combine_docs(self, docs):
        """
        Combines the content of multiple documents into a single string.

        Args:
            docs (list): A list of document objects, each having a 'page_content' attribute.

        Returns:
            str: A single string with the content of all documents separated by double newlines.
        """
        logging.info("Combining documents")
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_search(
        self,
        model_path="deepseek-r1:14b",
        query="Chris在AIF的角色是什麼？",
        db="./milvus_example.db",
        chunk_size=300,
        chunk_overlap=10,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Perform a Retrieval-Augmented Generation (RAG) search using a specified model and query.

        Args:
            model_path (str): Path to the language model to be used for generation.
            query (str): The query string to search for.
            db (str): Path to the Milvus database file.
            chunk_size (int): Size of text chunks for splitting documents.
            chunk_overlap (int): Overlap size between text chunks.
            model_name (str): Name of the model to be used for embedding.
            remove_think (bool): Flag to remove "think" from the response (default is False).

        Returns:
            None: Prints the response from the RAG chain.
        """
        # Embed text
        embedding = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "mps"}
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logging.info("Text splitting completed")

        all_splits = text_splitter.split_documents(self.pdf_reader)

        vector_store = Milvus(
            embedding_function=embedding,
            connection_args={"uri": db},
            # Set index_params if needed
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            auto_id=True,
        )
        vector_store.add_documents(all_splits)
        logging.info("Text embedding completed")
        # results = vector_store.similarity_search(query,k=1,)
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        # Convert the vector store to a retriever
        retriever = vector_store.as_retriever()
        logging.info("Retriever created")
        llm = OllamaLLM(model=model_path, base_url="http://localhost:11434")
        logging.info("LLM created")
        rag_chain = (
            {
                "context": retriever | self.combine_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logging.info("RAG chain created")
        # Invoke the RAG chain with a specific question and retrieve the response

        async def async_rag_search():
            async for chunk in rag_chain.astream(query):
                print(chunk, end="")

        asyncio.run(async_rag_search())


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_level=args.log_level)
    logging.info("Parsed command-line arguments")
    rag = MilvusRag(pdf_file=args.pdf_file)
    rag.rag_search(
        model_name=args.model_name,
        query=args.question,
        db=args.db_save_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_path=args.model_path,
    )
    logging.info("RAG process completed")
