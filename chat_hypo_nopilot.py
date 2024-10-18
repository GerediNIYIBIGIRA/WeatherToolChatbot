# import chainlit as cl
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from pyowm import OWM
# from langchain.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# import os
# from typing import Optional
# from langchain.tools import BaseTool

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OWM_API_KEY = os.getenv("OWM_API_KEY")

# # Set up GitHub loader
# loader = GithubFileLoader(
#     repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
#     access_token=ACCESS_TOKEN,
#     github_api_url="https://api.github.com",
#     file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
#     branch="main"
# )

# # Load documents
# documents = loader.load()

# # Set up embeddings and vector store
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(documents)
# vector = FAISS.from_documents(documents, embeddings)

# # Set up retriever tool
# retriever = vector.as_retriever()
# retriever_tool = create_retriever_tool(
#     retriever,
#     "malaria_search",
#     "Search for information about malaria. You must answer all questions about malaria according to the information that you were provided with",
# )

# # Set up Tavily search
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
# search = TavilySearchResults()

# class WeatherTool(BaseTool):
#     name: str = "WeatherTool"
#     description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country. Please check user's prompt if either country or city is not provided display an error message"

#     def _run(self, country: Optional[str] = None, city: Optional[str] = None) -> str:
#         owm = OWM(OWM_API_KEY)
#         mgr = owm.weather_manager()

#         if not city and not country:
#             return "Error: Both city and country are missing. Please provide both the city and the country."
#         elif not city:
#             return "Error: City is missing. Please provide both the city and the country."
#         elif not country:
#             return "Error: Country is missing. Please provide both the city and the country."
#         else:
#             try:
#                 location = f"{city},{country}"
#                 observation = mgr.weather_at_place(location)
#                 w = observation.weather
#                 temperature = w.temperature('celsius')['temp']
#                 status = w.detailed_status
#                 return f"The weather in {country}, {city} is {status} with a temperature of {temperature} degrees Celsius."
#             except Exception as e:
#                 return f"Error retrieving weather information: {e}"

#     async def _arun(self, country: Optional[str], city: Optional[str]) -> str:
#         raise NotImplementedError

# weather_tool = WeatherTool()

# tools = [retriever_tool, search, weather_tool]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria Prompt Answering Assistant for Geredi Niyibigira. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

# 1. Malaria-Related Queries:

# For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
# 2. Weather-Related Queries:

# If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
# If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
# If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
# If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
# 3. Unrelated Queries (e.g., Sports, Music, etc.):

# For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources.
# Clearly guide the user on where they can find appropriate information for these topics.
# Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# @cl.on_chat_start
# def setup_chain():
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#     llm_with_tools = llm.bind_tools(tools)
    
#     agent = (
#         {
#             "input": lambda x: x["input"],
#             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
#             "chat_history": lambda x: x["chat_history"]
#         }
#         | prompt
#         | llm_with_tools
#         | OpenAIToolsAgentOutputParser()
#     )
    
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     cl.user_session.set("agent_executor", agent_executor)


# @cl.on_message
# async def handle_message(message: cl.Message):
#     agent_executor = cl.user_session.get("agent_executor")
#     chat_history = cl.user_session.get("chat_history", [])
    
#     result = agent_executor.invoke({"input": message.content, "chat_history": chat_history})
    
#     chat_history.extend([
#         HumanMessage(content=message.content),
#         AIMessage(content=result["output"]),
#     ])
#     cl.user_session.set("chat_history", chat_history)
    
#     await cl.Message(content=result["output"]).send()

import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools import BaseTool
from pyowm import OWM
from langchain_community.document_loaders import GithubFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Optional

# Securely fetch API keys from environment variables
ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OWM_API_KEY = os.environ.get("OWM_API_KEY")

# Set up GitHub loader
loader = GithubFileLoader(
    repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
    access_token=ACCESS_TOKEN,
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
    branch="main"
)

# Load documents
documents = loader.load()

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(documents)
vector = FAISS.from_documents(documents, embeddings)

# Set up retriever tool
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "malaria_search",
    "Search for information about malaria. You must answer all questions about malaria according to the information that you were provided with",
)

# Set up Tavily search
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
search = TavilySearchResults()

class WeatherTool(BaseTool):
    name: str = "WeatherTool"
    description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country. Please check user's prompt if either country or city is not provided display an error message"

    def _run(self, country: Optional[str] = None, city: Optional[str] = None) -> str:
        owm = OWM(OWM_API_KEY)
        mgr = owm.weather_manager()

        if not city and not country:
            return "Error: Both city and country are missing. Please provide both the city and the country."
        elif not city:
            return "Error: City is missing. Please provide both the city and the country."
        elif not country:
            return "Error: Country is missing. Please provide both the city and the country."
        else:
            try:
                location = f"{city},{country}"
                observation = mgr.weather_at_place(location)
                w = observation.weather
                temperature = w.temperature('celsius')['temp']
                status = w.detailed_status
                return f"The weather in {city}, {country} is {status} with a temperature of {temperature} degrees Celsius."
            except Exception as e:
                return f"Error retrieving weather information: {e}"

    async def _arun(self, country: Optional[str], city: Optional[str]) -> str:
        raise NotImplementedError

weather_tool = WeatherTool()

tools = [retriever_tool, search, weather_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are the Malaria Prompt Answering Assistant for Geredi Niyibigira. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

1. Malaria-Related Queries:

For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
2. Weather-Related Queries:

If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
3. Unrelated Queries (e.g., Sports, Music, etc.):

For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources.
Clearly guide the user on where they can find appropriate information for these topics.
Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@cl.on_chat_start
def setup_chain():
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    llm_with_tools = llm.bind_tools(tools)
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def handle_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history", [])
    
    result = agent_executor.invoke({"input": message.content, "chat_history": chat_history})
    
    chat_history.extend([
        HumanMessage(content=message.content),
        AIMessage(content=result["output"]),
    ])
    cl.user_session.set("chat_history", chat_history)
    
    await cl.Message(content=result["output"]).send()
