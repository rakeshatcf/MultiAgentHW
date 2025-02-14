import os
from typing import Annotated

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, Cache, register_function
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
_ = load_dotenv()


def initialize_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my-docs-collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        print("Loading existing index...")
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    else:
        print("Creating new index...")
        documents = SimpleDirectoryReader("./createappdocs").load_data()
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )


index = initialize_index()
query_engine = index.as_query_engine()

def create_prompt(user_input):
    result = query_engine.query(user_input)

    prompt = f"""
    Your Task: Provide a concise and informative response to the user's query, drawing on the provided context.

    Context: {result}

    User Query: {user_input}

    Guidelines:
    1. Relevance: Focus directly on the user's question.   
    3. Accuracy: Ensure factual correctness.
    4. Clarity: Use clear language.
    5. Contextual Awareness: Use relevant knowledge from other projects if context is insufficient. 
    6. Honesty: State if you lack information.

    Response Format:
    - Direct answer
    - Detailed and complete explanation
    """
    print(prompt)
    return prompt

llm_config = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}
oai_config_list = {"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]}
from pathlib import Path
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
from autogen.coding import  LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
executor_docker = DockerCommandLineCodeExecutor(
    image="python:3.12-slim",  # Execute code using the given docker image name.
    timeout=60,  # Timeout for each code execution in seconds.
    work_dir=work_dir,  # Use the temporary directory to store the code files.
)
executor_cmdline = LocalCommandLineCodeExecutor(
    timeout=60,  # Timeout for each code execution in seconds.
    work_dir=work_dir,  # Use the temporary directory to store the code files.
)
# Create an agent with code executor configuration that uses docker.
code_executor_agent_using_docker = ConversableAgent(
    "code_executor_agent_docker",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor_cmdline},  # Use the docker command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
    is_termination_msg=lambda x: "content" in x
                                 and x["content"] is not None
                                 and x["content"].rstrip().endswith("TERMINATE"),
)
# code_writer_system_message = """You are a helpful AI assistant.
# You can use the task planner to decompose a complex task into sub-tasks.
# Make sure you follow through the sub-tasks.
# Solve task using your coding and language skills. Do not print any output on the screen.
# In the following cases, suggest python or react code (in a python or react coding block) or shell script (in a sh coding block) for the user to execute.
# 1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
# 2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
# Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Make assumptions if specifics are not clear. Be clear which step uses code, and which step uses your language skill.
# When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
# If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user. Write complete and executable code.
# If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
# When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
# Reply 'TERMINATE' in the end when everything is done.
# """
code_writer_system_message = """You are a helpful AI assistant.
You can use the task planner to decompose a complex task into sub-tasks.
Make sure you follow through the sub-tasks. Make assumptions when specific details to write code is not provided.
When needed, write Python code in markdown blocks, and I will execute them.
Give the user a final solution at the end.
Return TERMINATE only if the sub-tasks are completed.
"""
code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message=code_writer_system_message,
    llm_config=oai_config_list,
    code_execution_config=False,  # Turn off code execution for this agent.
)
planner = AssistantAgent(
    name="Project_planner",
    llm_config={
        "config_list": oai_config_list["config_list"],
        "cache_seed": None,  # Disable legacy cache.
    },
    system_message="You are a helpful AI assistant. Given a detailed project requirement scope you break them into modules. "
    "You only provide details for one module at a time. You can provide feasible implementation plan by breaking module into tasks . "
    "Please note that the information will all be retrieved using Python code. Please only suggest task implementation steps that can be retrieved using Python code."
)

# Create a planner user agent used to interact with the planner.
planner_user = UserProxyAgent(
    name="planner_user",
    human_input_mode="NEVER",
    code_execution_config=False,
)

def task_planner(question: Annotated[str, "Question to ask the planner."]) -> str:
    with Cache.disk(cache_seed=4) as cache:
        planner_user.initiate_chat(planner, message=question, max_turns=1, cache=cache)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="You are a RAG chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

register_function(
    task_planner,
    caller=code_writer_agent,
    executor=code_executor_agent_using_docker,
    name="task_planner",
    description="A task planner than can help you with decomposing a complex task into sub-tasks.",
)

def task_planner(question: Annotated[str, "Question to ask the planner."]) -> str:
    with Cache.disk(cache_seed=4) as cache:
        planner_user.initiate_chat(planner, message=question, max_turns=1, cache=cache)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

def main():
    #print("Welcome to RAGbot! Type 'exit', 'quit', or 'bye' to end the conversation.")
    # while True:
    #     user_input = input(f"\\nUser: ")
    #
    #     if user_input.lower() in ["exit", "quit", "bye"]:
    #         print(f"Goodbye! Have a great day!!")
    #         break

    prompt = create_prompt("Provide all the implementation details of all the modules in the project scope")

    reply = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])

    print(f"\\nRAGbot: {reply['content']}")
    # chat_result = code_executor_agent_using_docker.initiate_chat(
    #     code_writer_agent,
    #     message=f"Write Python code to write an application that implements the module details given from a business requirements document. {reply['content']}",
    # )
    with Cache.disk(cache_seed=1) as cache:
        # the assistant receives a message from the user, which contains the task description
        code_executor_agent_using_docker.initiate_chat(
            code_writer_agent,
            message=f"Get implementation plan for one module at a time from the project planner: {reply['content']}",
            cache=cache,
        )
if __name__ == "__main__":
    main()