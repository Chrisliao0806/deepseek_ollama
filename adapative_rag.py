"""Langgraph adaptive RAG system."""

import argparse
import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from utils.logger import setup_logging
from langchain.callbacks import get_openai_callback

# Prompt Template for RAG
INSTRUCTIONRAG = """
你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。
注意：請確保答案的準確性。
"""

# Prompt Template for PLAIN
INSTRUCTIONPLAIN = """
你是一位負責處理使用者問題的助手，請利用你的知識來回應問題。
回應問題時請確保答案的準確性，勿虛構答案。
"""


def parse_arguments():
    """
    Parses command-line arguments for the RAG model using langgraph.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Arguments:
        --pdf-file (str): The path to the PDF file. Default is "Chris_Resume.pdf".
        --model-path (str): The path to the ollama file. Default is "qwen2.5:7b".
        --chunk-size (int): The maximum size of each text chunk. Default is 300.
        --chunk-overlap (int): The number of characters that overlap between chunks. Default is 10.
        --model-name (str): The name of the model to use for embedding.
                            Default is "sentence-transformers/all-MiniLM-L6-v2".
        --log-level (str): The logging level. Default is "INFO".
        --question (str): The question to ask the model. Default is "可以幫我看一下po yu liao的職業嗎".
        --save-img (bool): If True, saves the mermaid diagram as an image. Default is True.
    """
    parser = argparse.ArgumentParser(description="RAG model for using langgraph")
    parser.add_argument(
        "--pdf-file",
        default="Chris_Resume.pdf",
        type=str,
        help="The path to the PDF file.",
    )
    parser.add_argument(
        "--model-path",
        default="qwen2.5:7b",
        type=str,
        help="The path to the ollama file.",
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
        "--log-level",
        default="INFO",
        type=str,
        help="The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--question",
        default="可以幫我看一下po yu liao的職業嗎",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--save-img",
        default=True,
        type=bool,
        help="If True, saves the mermaid diagram as an image.",
    )
    return parser.parse_args()


class GraphState(TypedDict):
    """
    State of graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    use_rag: bool


class AdaptiveRag:
    """
    AdaptiveRag is a class that implements a retrieval-augmented generation (RAG) system for processing and answering questions
    based on a given PDF document. It uses a combination of document embedding, text splitting, and a language model to generate responses.
    Attributes:
        model_name (str): The name of the model used for embeddings.
        chunk_size (int): The size of text chunks for splitting the document.
        chunk_overlap (int): The overlap size between text chunks.
        pdf_reader (PyMuPDFLoader): The PDF reader object for loading the document.
        llm (ChatOllama): The language model used for generating responses.
        rag_chain (Chain): The chain of operations for RAG mode.
        llm_chain (Chain): The chain of operations for plain LLM mode.
        vectordb (Chroma): The vector database for storing document embeddings.
        retriever (Retriever): The retriever object for searching relevant documents.
    """

    def __init__(
        self,
        pdf_file: str,
        chunk_size: int,
        chunk_overlap: int,
        model_path: str,
        model_name: str,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_reader = PyMuPDFLoader(pdf_file).load()
        self.document_embedding()
        self.llm = ChatOllama(model=model_path, base_url="http://localhost:11434")
        self.rag_chain, self.llm_chain = self._init_model()

    def _init_model(self):
        prompt_rag = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAG),
                ("system", "文件: \n\n {documents}"),
                ("human", "問題: {question}"),
            ]
        )

        prompt_plain = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONPLAIN),
                ("human", "問題: {question}"),
            ]
        )
        # LLM & chain
        rag_chain = prompt_rag | self.llm | StrOutputParser()
        # LLM & chain
        llm_chain = prompt_plain | self.llm | StrOutputParser()
        return rag_chain, llm_chain

    def document_embedding(self):
        """
        Generates document embeddings and initializes a vector database for retrieval.

        This method performs the following steps:
        1. Embeds the text using a HuggingFace model.
        2. Splits the text into chunks using a RecursiveCharacterTextSplitter.
        3. Creates a Chroma vector database from the split documents.
        4. Initializes a retriever from the vector database for document retrieval.

        Attributes:
            model_name (str): The name of the HuggingFace model to use for embedding.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between text chunks.
            pdf_reader (object): The PDF reader object containing the documents to be split.
            vectordb (Chroma): The Chroma vector database created from the documents.
            retriever (object): The retriever initialized from the vector database.

        Returns:
            None
        """
        # Embed text
        embedding = HuggingFaceEmbeddings(
            model_name=self.model_name, model_kwargs={"device": "mps"}
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        logging.info("Text splitting completed")

        all_splits = text_splitter.split_documents(self.pdf_reader)
        self.vectordb = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding,
            collection_name="coll2",
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, state):
        """
        Retrieve relevant documents based on the given question in the state.

        Args:
            state (dict): A dictionary containing the question to be used for retrieval.

        Returns:
            dict: A dictionary containing:
                - "documents" (list): A list of tuples where each tuple
                                      contains a document and its relevance score.
                - "question" (str): The original question from the state.
                - "use_rag" (bool): A flag indicating whether the relevance score
                                    of any document exceeds the threshold (0.3).
        """

        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        # 0.3 is the threshold for relevance score
        documents = self.vectordb.similarity_search_with_relevance_scores(question)
        for res, score in documents:
            print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
            if score > 0.3:
                return {"documents": documents, "question": question, "use_rag": True}
        return {"documents": documents, "question": question, "use_rag": False}

    def route_retrieve(self, state):
        """
        Determines the retrieval route based on the state configuration.

        Args:
            state (dict): A dictionary containing the state configuration.
                          It must include a key "use_rag" with a boolean value.

        Returns:
            str: Returns "rag_generate" if "use_rag" is True, otherwise returns "plain_answer".
        """
        print("---ROUTE RETRIEVE---")
        use_rag = state["use_rag"]
        if use_rag:
            print("  -ROUTE TO RAG-")
            return "rag_generate"
        else:
            print("  -ROUTE TO PLAIN FEEDBACK-")
            return "plain_answer"

    def rag_generate(self, state):
        """
        Generates a response in RAG (Retrieval-Augmented Generation) mode.

        This method takes a state dictionary containing a question and a list of documents,
        and uses the RAG chain to generate a response based on the provided documents and question.

        Args:
            state (dict): A dictionary containing the following keys:
                - "question" (str): The question to be answered.
                - "documents" (list): A list of documents to be used for generating the response.

        Returns:
            dict: A dictionary containing the original question,
                  documents, and the generated response.
                - "question" (str): The original question.
                - "documents" (list): The original list of documents.
                - "generation" (str): The generated response.
        """
        print("---GENERATE IN RAG MODE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def plain_answer(self, state):
        """
        Generate answer using the LLM without vectorstore.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        print("---GENERATE PLAIN ANSWER---")
        question = state["question"]
        generation = self.llm_chain.invoke({"question": question})
        return {"question": question, "generation": generation}

    def workflow_setup(self, question):
        """
        Sets up and compiles a workflow for processing a given question.

        This method initializes a StateGraph with nodes for different processing steps
        (RAG generation, plain answer generation, and retrieval). It defines the edges
        and conditional transitions between these nodes, compiles the workflow, and
        invokes it with the provided question.

        Args:
            question (str): The question to be processed by the workflow.

        Returns:
            tuple: A tuple containing the compiled workflow and the result of invoking
                   the workflow with the given question.
        """
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("rag_generate", self.rag_generate)  # rag
        workflow.add_node("plain_answer", self.plain_answer)  # llm
        workflow.add_node("retrieve", self.retrieve)  # retrieve

        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.route_retrieve,
            {"rag_generate": "rag_generate", "plain_answer": "plain_answer"},
        )
        workflow.add_edge("rag_generate", END)
        workflow.add_edge("plain_answer", END)

        # Compile
        compiled_app = workflow.compile()
        with get_openai_callback() as cb:
             output = compiled_app.invoke({"question": question})
             print(f"Total Tokens: {cb.total_tokens}")
             print(f"input_tokens: {cb.prompt_tokens}")
             print(f"output_tokens: {cb.completion_tokens}")


        return compiled_app, compiled_app.invoke({"question": question})


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_level=args.log_level)
    adaptive_rag = AdaptiveRag(
        pdf_file=args.pdf_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_path=args.model_path,
        model_name=args.model_name,
    )
    app, output = adaptive_rag.workflow_setup(question=args.question)
    print(output["generation"])

    if args.save_img:
        image_data = app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
        FILE = "mermaid_diagram.png"
        try:
            with open(FILE, "wb") as f:
                f.write(image_data)
            print(f"image save: {FILE}")
        except IOError as e:
            print(f"image save error: {e}")
