import argparse
import asyncio
from utils.logger import setup_logging
from utils.llm_usage import ollama_llm


def parse_arguments():
    """
    Parses command-line arguments for the DeepSeek-R1 model using langchain.

    Returns:
        argparse.Namespace: A namespace containing the parsed command-line arguments:
            - model_path (str): The path to the DeepSeek-R1 model file. Default is "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf".
            - question (str): The question to ask the model. Default is "｜User｜>What is 1+1?<｜Assistant｜>".
            - nctx (int): The number of tokens to output for context. Default is 512.
            - max_tokens (int): The maximum number of tokens to generate. Default is 512.
    """
    parser = argparse.ArgumentParser(
        description="deepseek r1 model for using langchain"
    )
    parser.add_argument(
        "--question",
        default="What is 1+1?",
        type=str,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--model-path",
        default="deepseek-r1:14b",
        type=str,
        help="The path to the DeepSeek-R1 model file.",
    )
    parser.add_argument(
        "--remove-think",
        default=False,
        type=bool,
        help="If True, removes content between <think> and </think> tags from the response. Defaults to False.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging("INFO")

    print(
        asyncio.run(
            ollama_llm(
                question=args.question,
                model_name=args.model_path,
                remove_think=args.remove_think,
            )
        )
    )
