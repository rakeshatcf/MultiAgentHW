import os
from pathlib import Path
import chromadb
from typing import Annotated
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
_ = load_dotenv()
llm_config={"model": "gpt-4o"}

from autogen.coding import LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
# import chromadb
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
# chroma_client.list_collections()

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

executor = DockerCommandLineCodeExecutor(work_dir=work_dir)
from autogen import ConversableAgent, AssistantAgent
termination_msg = "TERMINATE"
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)
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

llm_config_groq = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY"),
            "api_type": "groq",
        }
    ]
}

boss = autogen.UserProxyAgent(
    name="Boss",
    #is_termination_msg=termination_msg,
    human_input_mode="TERMINATE",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False
)

rag_agent = ConversableAgent(
    name="RAGbot",
    system_message="You are a RAG chatbot",
    llm_config=llm_config_groq,
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder = autogen.AssistantAgent(
    name="Senior_Python_Engineer",
    #is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    #is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    #is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

def retrieve_content(
    message: Annotated[
        str,
        "Refined message which keeps the original meaning and can be used to retrieve content for code generation and execution.",
    ],
    #n_results: Annotated[int, "number of results"] = 3,
) -> str:
    #boss_aid.n_results = n_results  # Set the number of results to be retrieved.
    #_context = {"problem": message, "n_results": n_results}
    prompt = create_prompt(message)
    ret_msg = rag_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
    return ret_msg['content'] or message

for caller in [pm, coder, reviewer]:
    d_retrieve_content = caller.register_for_llm(
        description="retrieve content for code generation and execution.", api_style="function"
    )(retrieve_content)

for executor in [boss, pm, code_executor_agent]:
    executor.register_for_execution()(d_retrieve_content)

groupchat = autogen.GroupChat(
    agents=[boss, pm, coder, reviewer, code_executor_agent],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)

#llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0}
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Start chatting with the boss as this is the user proxy agent.
boss.initiate_chat(
    manager,
    message="Create a working application using all details of the project. Don't omit anything",
)
