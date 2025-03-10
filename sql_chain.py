import argparse
import logging
from typing_extensions import TypedDict

from typing_extensions import Annotated
from langchain_ollama import ChatOllama
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_community.utilities import SQLDatabase
from langgraph.graph import START, StateGraph
from utils.logger import setup_logging
from langchain.callbacks import get_openai_callback


class State(TypedDict):
    """State for the SQL retrieval process
    question: The question to ask the model.
    query: The generated SQL query.
    result: The result of the SQL query.
    answer: The answer to the question.
    """

    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def parse_arguments():
    """
    Parses command-line arguments for the sql chain model.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Arguments:
        --log-level (str): The logging level (default: "INFO"). Options are DEBUG, INFO, WARNING, ERROR, CRITICAL.
        --question (str): The question to ask the model (default: "可以幫我從Issue_Header找出最常見的issue嗎？").
        --db-uri (str): The URI of the database (default: "sqlite:///Innodisk.db").
        --model (str): The name of the model to use for embedding (default: "qwen2.5:7b").
    """
    parser = argparse.ArgumentParser(description="sql chain model for using langgraph")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help="The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--question",
        default="可以幫我從Issue_Header找出最常見的issue嗎？",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--db-uri",
        default="sqlite:///Innodisk.db",
        type=str,
        help="The URI of the database.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        type=str,
        help="The name of the model to use for embedding.",
    )

    return parser.parse_args()


class SqlRetrieve:
    """
    Attributes:
        llm (ChatOllama): The language model used for generating SQL queries and answers.
        query_prompt_template (PromptTemplate): The template for generating SQL query prompts.
        db (SQLDatabase): The database connection object.

    Methods:
        __init__(db_uri, model="qwen2.5:7b"):
            Initializes the SqlRetrieve object with the given database URI and model.

        _db_query(query="SELECT Subject FROM Issue_Header GROUP BY Subject ORDER BY COUNT(*) DESC LIMIT 25"):
            Executes a database query and prints the results.

        _show_prompt():

        write_query(state: State) -> dict:

        execute_query(state: State) -> dict:
            Executes the SQL query stored in the state and returns the result.

        generate_answer(state: State) -> dict:
            Generates an answer to the user's question using the retrieved information as context.

        workflow(query: str):
            Defines the workflow for the SQL retrieval process, including writing the query,
            executing the query, and generating the answer.
    """

    def __init__(
        self,
        db_uri,
        model="qwen2.5:7b",
    ):
        self.llm = ChatOllama(model=model, base_url="http://localhost:11434")
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        self.db = SQLDatabase.from_uri(db_uri)

    def _db_query(self, query: str):
        """
        Executes a database query and prints the results.
        """
        print(self.db.dialect)
        print(self.db.get_usable_table_names())
        print(self.db.run(query))

    def _show_prompt(self):
        """
        Displays the prompt message in a formatted manner.

        This method asserts that there is exactly one message in the
        query_prompt_template's messages list and then prints it
        using the pretty_print method.
        """
        assert len(self.query_prompt_template.messages) == 1
        self.query_prompt_template.messages[0].pretty_print()

    def write_query(self, state: State):
        """
        Generates and executes a query based on the provided state.

        Args:
            state (State): The current state containing the question to be converted into a query.

        Returns:
            dict: A dictionary containing the generated query under the key "query".
        """
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 25,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        print(result)
        return {"query": result["query"]}

    def execute_query(self, state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(self, state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}"
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def workflow(self, query: str):
        """
        Define the workflow for the SQL retrieval process.

        This method constructs a state graph for the SQL retrieval process,
        which includes writing the query, executing the query, and generating
        the answer. It then streams the steps of the graph while printing
        the progress and token usage.

        Args:
            query (str): The SQL query to be processed.

        Returns:
            None
        """
        token = []
        graph_builder = StateGraph(State).add_sequence(
            [self.write_query, self.execute_query, self.generate_answer]
        )
        graph_builder.add_edge(START, "write_query")
        graph = graph_builder.compile()
        with get_openai_callback() as cb:
            for step in graph.stream(
                {"question": query},
                stream_mode="updates",
            ):
                print(step)
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"input_tokens: {cb.prompt_tokens}")
                print(f"output_tokens: {cb.completion_tokens}")
            token.append(cb.total_tokens)
            token.append(cb.prompt_tokens)
            token.append(cb.completion_tokens)

        return step["generate_answer"]["answer"], token


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(log_level=args.log_level)
    logging.info("Parsed command-line arguments")
    sql_retrieve = SqlRetrieve(
        db_uri=args.db_uri,
        model=args.model,
    )
    output_answer, token_output = sql_retrieve.workflow(query=args.question)
    print(output_answer)
    print(token_output)
    logging.info("SQL process completed")
