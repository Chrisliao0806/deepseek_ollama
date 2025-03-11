"""Langgraph adaptive RAG system."""

import os
import argparse
import logging
from typing import List
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict

from utils.logger import setup_logging
from langchain.callbacks import get_openai_callback


load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class RAGState(BaseModel):
    """
    向量資料庫回覆工具。若問題可以從向量資料庫中找到答案，則使用RAG工具回覆。
    """

    query: str = Field(description="使用向量資料庫回覆時輸入的問題")


class PlainState(BaseModel):
    """
    直接回覆工具。若問題從向量資料庫中找不到的話，則直接用自己的知識進行回覆
    """

    query: str = Field(description="使用直接回覆時輸入的問題")


class WebState(BaseModel):
    """
    網路搜尋工具。若問題覺得需要用網路查詢，則使用WebState工具搜尋解答。
    """

    query: str = Field(description="使用網路搜尋時輸入的問題")


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

# Prompt Template for WEB
INSTRUCTIONWEB = """
你是一位負責處理使用者問題的助手，使用者會輸入一個問題，\
你的目標是要去確認這個問題到底是在跟你聊天還是想要詢問你事情。

如果使用者的問題就是純粹的聊天，則使用PlainState工具(只有聊天才用此工具)
如果使用者的問題不是純粹的聊天，而是詢問你事情的話，則使用WebState工具
"""

# Prompt Template for RAG
INSTRUCTIONWEBRAG = """
你是一位負責處理使用者問題的助手，請利用提取出來的網頁內容來回應問題。
你必須從網頁內容提取出答案，並回答使用者的問題。
注意：請確保答案的準確性。並且不能回答出跟網頁不一樣的資訊出來
"""

INSTRUCTIONRAGORPLAIN = """
你是一位負責處理使用者問題的助手，使用者會輸入一個問題，然後上述會有一個文件的內容，\
你的目標就是去確認這個問題是否可以從這個文件中找到答案。
你要能判斷問題本身，如：可以介紹這篇文章或是這篇文章有什麼特別的地方等等，這都算是RAGState工具的範疇。

如果文件裡面的內容與使用者問題有關聯，就要使用 RAGState 工具。
最後如果文件裡面的內容與使用者問題完全無關，請使用 PlainState 工具。
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
        (
            self.rag_chain,
            self.llm_chain,
            self.web_chain,
            self.question_router,
            self.question_router_rag_or_plain,
        ) = self._init_model()
        self.web_search_tool = TavilySearchResults(
            include_answer=True,
        )

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

        prompt_web = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONWEBRAG),
                ("human", "問題: {question}"),
                ("system", "網頁內容: \n\n {documents}"),
            ]
        )
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONWEB),
                ("human", "問題: {question}"),
            ]
        )
        route_prompt_rag_or_plain = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONRAGORPLAIN),
                ("system", "文件: \n\n {documents}"),
                ("human", "問題: {question}"),
            ]
        )

        # LLM & chain
        rag_chain = prompt_rag | self.llm | StrOutputParser()
        # LLM & chain
        llm_chain = prompt_plain | self.llm | StrOutputParser()
        # LLM & chain
        web_chain = prompt_web | self.llm | StrOutputParser()
        # Route LLM with tools use
        structured_llm_router = self.llm.bind_tools(tools=[WebState, PlainState])
        question_router = route_prompt | structured_llm_router
        # Route LLM with tools use
        structured_rag_plain_router = self.llm.bind_tools(tools=[RAGState, PlainState])
        question_router_rag_or_plain = (
            route_prompt_rag_or_plain | structured_rag_plain_router
        )

        return (
            rag_chain,
            llm_chain,
            web_chain,
            question_router,
            question_router_rag_or_plain,
        )

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
        # self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

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
        print(documents)
        return {"documents": documents, "question": question, "use_rag": False}

    ### Edges ###
    def route_rag_plain_test(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        documents = state["documents"]
        source = self.question_router_rag_or_plain.invoke(
            {"question": question, "documents": documents}
        )

        if len(source.tool_calls) == 0:
            print("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"

        if source.tool_calls[0]["name"] == "RAGState":
            print("  -ROUTE TO RAG-")
            return "rag_generate"
        else:
            print("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"

    ### Edges ###
    def route_web_test(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        print(source)
        if len(source.tool_calls) == 0:
            print("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"
        
        if source.tool_calls[0]["name"] == "WebState":
            print("  -ROUTE TO WEB SEARCH-")
            return "web_search"
        
        elif source.tool_calls[0]["name"] == "PlainState":
            print("  -ROUTE TO PLAIN LLM-")
            return "plain_feedback"

        # else:
        #     print("  -ROUTE TO WEB SEARCH-")
        #     return "web_search"

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

    def web_generate(self, state):
        """
        Generates a response using web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains web search generation
        """
        print("---WEB GENERATE---")
        question = state["question"]
        documents = self.web_search_tool.invoke({"query": question})
        documents = [doc["content"] for doc in documents]
        print(documents)
        # RAG generation
        generation = self.web_chain.invoke(
            {"documents": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

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

    def first_stage_end(self, state):
        """
        End of the first stage.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        print("---FIRST STAGE END---")
        return {"question": state["question"]}

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
        workflow.add_node("web_search", self.web_generate)  # web search
        workflow.add_node("first_stage_end", self.first_stage_end)  # end of first stage

        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.route_rag_plain_test,
            {"rag_generate": "rag_generate", "plain_feedback": "first_stage_end"},
        )
        workflow.add_conditional_edges(
            "first_stage_end",
            self.route_web_test,
            {"web_search": "web_search", "plain_feedback": "plain_answer"},
        )
        workflow.add_edge("rag_generate", END)
        workflow.add_edge("web_search", END)
        workflow.add_edge("plain_answer", END)

        # Compile
        compiled_app = workflow.compile()
        with get_openai_callback() as cb:
            output = compiled_app.invoke({"question": question})
            print(output["generation"])
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"input_tokens: {cb.prompt_tokens}")
            print(f"output_tokens: {cb.completion_tokens}")

        return compiled_app


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
    app = adaptive_rag.workflow_setup(question=args.question)

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
