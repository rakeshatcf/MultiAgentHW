import os
from typing import Annotated

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, Cache, register_function, GroupChat, \
    GroupChatManager, Agent
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

llama_llm_config = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}
oai_config_list = {"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}
from pathlib import Path
work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)
from autogen.coding import  LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
executor_docker = DockerCommandLineCodeExecutor(
    image="broadinstitute/python-requests",#"python:3.12-alpine",  # Execute code using the given docker image name.
    timeout=60,  # Timeout for each code execution in seconds.
    work_dir=work_dir,  # Use the temporary directory to store the code files.
)
executor_cmdline = LocalCommandLineCodeExecutor(
    timeout=60,  # Timeout for each code execution in seconds.
    work_dir=work_dir,  # Use the temporary directory to store the code files.
)
# Create an agent with code executor configuration that uses docker.
code_executor_agent = ConversableAgent(
    "code_executor_agent",
    #llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor_docker},  # Use the docker command line code executor.#, "last_n_messages":3
    description = "Executor should always be called after the engineer has written code to be executed.",
    human_input_mode="TERMINATE",  # Always take human input for this agent for safety.
    is_termination_msg=lambda x: "content" in x
                                 and x["content"] is not None
                                 and x["content"].rstrip().endswith("TERMINATE"),
)


code_writer_system_message = """Senior Engineer. You take a USER STORY and implement it. You write python/bash to implement tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor. Do not write any code that requires user input. Generate and use sample data as default for any keys, IDs, table names or any other input.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. 
suggest python code (in a python block) or shell script (in a sh coding block) for the user to execute.
 1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
 2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
 Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Make assumptions if specifics are not clear. Be clear which step uses code, and which step uses your language skill.
 When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
 If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result.  Check the execution result returned by the user. Write complete and executable code.
 Use minimum library dependencies necessary to write the code. Accept 'exitcode: 0' as successful execution and continue.
 If the response is more than 8000 tokens then only return the last 8000 tokens.
 If a user story acceptance criteria has been met by code execution then return 'DONE'
"""
code_writer_agent = ConversableAgent(
    "Senior_Engineer",
    system_message=code_writer_system_message,
    llm_config=oai_config_list,
    code_execution_config=False,  # Turn off code execution for this agent.
)
# planner = AssistantAgent(
#     name="Project_planner",
#     llm_config={
#         "config_list": oai_config_list["config_list"],
#         "cache_seed": None,  # Disable legacy cache.
#     },
#     system_message="You are a helpful AI assistant. Given a detailed project requirement scope you break them into modules. "
#     "You only provide details for one module at a time. You can provide feasible implementation plan by breaking module into tasks . "
#     "Please note that the information will all be retrieved using Python code. Please only suggest task implementation steps that can be retrieved using Python code."
# )

# Create a planner user agent used to interact with the planner.
user_proxy = UserProxyAgent(
    name="Client",
    system_message="A client who has a detailed description of project requirements. The detailed project requirements are sent do product owner one feature at a time with project summary",
    code_execution_config=False,
    human_input_mode="NEVER",
)

product_owner =  ConversableAgent(
    name="Product_Owner",
    system_message = """
    You are PRODUCT OWNER and an expert of software development agency with 20 years of experience with excellent skills of describing information and structuring. 
    You also have extraordinary skill in business and system analysis. All your answers are always well structured.
    Your main task is to break the feature down into USER STORIES. You can only create up to 10 USER STORIES per feature. Provide well-structured USER STORY based on best industry practices. 
    If you need additional information to write the story then make any assumptions necessary to provide all necessary implementation details.
    The whole user story should be based on the SMART approach. The result of the story should be to achieve the goals of the feature at hand.
    When your turn comes respond with only one USER STORY at a time until all stories are implemented. You can skip repository setup and CI/CD implementation features.
    STRUCTURE OF USER STORY

    Title of the story.
    
    Users story — max 25 words. Use format: “as a [ROLE], I want [WHAT SHOULD BE DONE], so that [VALUE ROLE GAIN]”
    
    Importance of these user story. How it will help to achieve goals of EPIC and FEATURE.
    
    Context of the story. well restructured with fixed typos and grammar mistakes. Shortened version of user inputs.
    
    Recommended steps. 
     
    Acceptance criteria — build it based on best industry practices, it should be according to the SMART framework.
    
    """,
    llm_config = oai_config_list,
    code_execution_config = False,  # Turn off code execution for this agent.
)

# scrum_master = AssistantAgent(
#     name="Scrum_Master",
#     system_message="""Software Development Manager. Given a project requirements task, please determine what information is needed to implement the task. Skip any repository setup and CI/CD implementation tasks. Only focus on one task at a time. If additional information is required then make sample data and take a decision. Do not ask for more information.
# """,
#     llm_config={"config_list": oai_config_list["config_list"], "cache_seed": None},
# )


rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="You are a RAG chatbot",
    llm_config=llama_llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages

    if len(messages) <= 1:
        # first, let the engineer retrieve relevant data
        return product_owner

    if last_speaker is product_owner:
        # if the last message is from planner, let the engineer to write code
        return code_writer_agent
    elif last_speaker is user_proxy:
        if messages[-1]["content"].strip() != "":
            # If the last message is from user and is not empty, let the writer to continue
            return product_owner

    elif last_speaker is code_writer_agent:
        if "```python" or "```sh" or "```jsx" in messages[-1]["content"]:
            # If the last message is a python code block, let the executor to speak
            return code_executor_agent
        else:
            # Otherwise, let the engineer to continue
            return code_writer_agent

    elif last_speaker is code_executor_agent:
        if "exitcode: 1" in messages[-1]["content"] or "DONE" not in messages[-1]["content"]:
            # If the last message indicates an error, let the engineer to improve the code
            return code_writer_agent
        else:
            # Otherwise, let the writer to speak
            return product_owner

    # elif last_speaker is writer:
    #     # Always let the user to speak after the writer
    #     return user_proxy

    else:
        # default to auto speaker selection method
        return "auto"

groupchat = GroupChat(
    agents=[user_proxy, product_owner, code_writer_agent, code_executor_agent],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": oai_config_list["config_list"], "cache_seed": None})

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
    with Cache.disk(cache_seed=41) as cache:
        # the assistant receives a message from the user, which contains the task description
        user_proxy.initiate_chat(
            manager,
            message=f"Use the provided project requirements : {reply['content']}",
            cache=cache,
        )
if __name__ == "__main__":
    main()