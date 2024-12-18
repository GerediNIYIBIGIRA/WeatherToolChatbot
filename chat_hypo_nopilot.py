

# import os
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
# from langchain_community.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from typing import Optional

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# OWM_API_KEY = os.environ.get("OWM_API_KEY")

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
#                 return f"The weather in {city}, {country} is {status} with a temperature of {temperature} degrees Celsius."
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

# import os
# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from pyowm import OWM
# from langchain_community.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OWM_API_KEY = os.getenv("OWM_API_KEY")

# # Streamlit page configuration
# st.set_page_config(page_title="Geredi AI Malaria and Weather Tool Assistant", page_icon="🦟", layout="wide")

# # Custom CSS for styling to match the logo's aesthetic
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #ffffff;
#         color: #000000;
#     }
#     .main-header {
#         font-size: 2.5rem;
#         color: #000000;
#         text-align: center;
#         padding: 1rem 0;
#         font-weight: bold;
#         font-family: 'Arial', sans-serif;
#     }
#     .chat-container {
#         background-color: #f7f7f7;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .human-message {
#         background-color: #e0e0e0;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#     }
#     .ai-message {
#         background-color: #d4d4d4;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#     }
#     .copyright {
#         text-align: center;
#         margin-top: 20px;
#         font-size: 0.8rem;
#         color: #333333;
#     }
#     .stTextInput > div > div > input {
#         color: #000000;
#         background-color: #ffffff;
#         border: 1px solid #cccccc;
#     }
#     .stButton > button {
#         color: #ffffff;
#         background-color: #000000;
#         border: none;
#     }
#     .stButton > button:hover {
#         color: #ffffff;
#         background-color: #333333;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Streamlit app title
# st.markdown("<h1 class='main-header'>Malaria and Weather Tool Assistant</h1>", unsafe_allow_html=True)

# # Load documents from GitHub
# loader = GithubFileLoader(
#     repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
#     access_token=ACCESS_TOKEN,
#     github_api_url="https://api.github.com",
#     file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
#     branch="main"
# )
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
#     description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country."

#     def _run(self, country: str = None, city: str = None) -> str:
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

#     async def _arun(self, country: str, city: str) -> str:
#         raise NotImplementedError

# weather_tool = WeatherTool()

# tools = [retriever_tool, search, weather_tool]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria Prompt Answering Assistant developed by Geredi Niyibigira if someone or user a question who developed you or who manage you please answer him/her that you have developed and managed by Geredi Niyibigira a graduated student in MS in Engineering Artificial Intelligence at Carnegie Mellon University Africa. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

# 1. Malaria-Related Queries:
#    For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
# 2. Weather-Related Queries:
#    If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
#    If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
#    If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
#    If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
# 3. Unrelated Queries (e.g., Sports, Music, etc.):
#    For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources.
#    Clearly guide the user on where they can find appropriate information for these topics.
#    Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Initialize session state to store chat history and agent executor
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "agent_executor" not in st.session_state:
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
    
#     st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Streamlit input for user message
# st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
# user_input = st.text_input("Welcome to Geredi AI! I'm here to assist you with any questions or information related to malaria and weather: ", "")

# if st.button("Send"):
#     if user_input:
#         agent_executor = st.session_state.agent_executor
#         chat_history = st.session_state.chat_history

#         result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

#         # Update chat history
#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=result["output"]))
#         st.session_state.chat_history = chat_history

#         # Display the conversation
#         for message in chat_history:
#             if isinstance(message, HumanMessage):
#                 st.markdown(f"<div class='human-message'><strong>Human:</strong> {message.content}</div>", unsafe_allow_html=True)
#             elif isinstance(message, AIMessage):
#                 st.markdown(f"<div class='ai-message'><strong>Geredi AI:</strong> {message.content}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='ai-message'><strong>{type(message).__name__}:</strong> {message.content}</div>", unsafe_allow_html=True)
#     else:
#         st.warning("Please enter a message before sending.")

# st.markdown("</div>", unsafe_allow_html=True)

# # Copyright notice
# st.markdown("<p class='copyright'>© 2024 Developed and Managed by Geredi NIYIBIGIRA. All rights reserved.</p>", unsafe_allow_html=True)
# #############################################################################################################################################

# import os
# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from pyowm import OWM
# from langchain_community.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OWM_API_KEY = os.getenv("OWM_API_KEY")

# # Streamlit page configuration
# st.set_page_config(page_title="Geredi AI Malaria and Weather Tool Assistant", page_icon="🦟", layout="wide")

# # Custom CSS for styling to match the logo's aesthetic and improve UI
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #F5F5F5;
#         color: #333333;
#     }
#     .main-header {
#         font-size: 3rem;
#         color: #0E5A8A;
#         text-align: center;
#         padding: 1.5rem 0;
#         font-weight: bold;
#         font-family: 'Verdana', sans-serif;
#     }
#     .chat-container {
#         background-color: #ffffff;
#         border-radius: 10px;
#         padding: 25px;
#         box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
#     }
#     .human-message, .ai-message {
#         padding: 15px;
#         border-radius: 10px;
#         margin: 8px 0;
#         font-size: 1rem;
#     }
#     .human-message {
#         background-color: #e0e0e0;
#         border-left: 5px solid #0E5A8A;
#     }
#     .ai-message {
#         background-color: #e4eff1;
#         border-left: 5px solid #FF6347;
#     }
#     .stTextInput > div > div > input {
#         border: 2px solid #333333;
#         padding: 10px;
#         font-size: 1rem;
#     }
#     .stButton > button {
#         background-color: #0E5A8A;
#         color: #ffffff;
#         padding: 0.8rem;
#         font-size: 1rem;
#         border-radius: 5px;
#     }
#     .stButton > button:hover {
#         background-color: #0D4F6A;
#     }
#     .copyright {
#         text-align: center;
#         margin-top: 20px;
#         font-size: 0.85rem;
#         color: #666666;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Streamlit app title
# st.markdown("<h1 class='main-header'>Malaria and Weather Tool Assistant</h1>", unsafe_allow_html=True)

# # Load documents from GitHub
# loader = GithubFileLoader(
#     repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
#     access_token=ACCESS_TOKEN,
#     github_api_url="https://api.github.com",
#     file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
#     branch="main"
# )
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
#     description: str = "Useful for getting the weather in a specific location. Input should be a city and country."

#     def _run(self, country: str = None, city: str = None) -> str:
#         owm = OWM(OWM_API_KEY)
#         mgr = owm.weather_manager()

#         if not city and not country:
#             return "Error: Both city and country are missing. Please provide both."
#         elif not city:
#             return "Error: City is missing. Please provide both city and country."
#         elif not country:
#             return "Error: Country is missing. Please provide both city and country."
#         else:
#             try:
#                 location = f"{city},{country}"
#                 observation = mgr.weather_at_place(location)
#                 w = observation.weather
#                 temperature = w.temperature('celsius')['temp']
#                 status = w.detailed_status
#                 return f"The weather in {city}, {country} is {status} with a temperature of {temperature}°C."
#             except Exception as e:
#                 return f"Error retrieving weather information: {e}"

#     async def _arun(self, country: str, city: str) -> str:
#         raise NotImplementedError

# weather_tool = WeatherTool()

# tools = [retriever_tool, search, weather_tool]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria and Weather Assistant developed by Geredi Niyibigira. Your role is to provide answers about malaria and current weather information in a friendly and professional manner. For malaria queries, use RAG content. For weather, retrieve real-time data.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Initialize session state to store chat history and agent executor
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "agent_executor" not in st.session_state:
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
    
#     st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Streamlit input for user message
# st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
# user_input = st.text_input("Ask me about malaria or weather: ", "")

# if st.button("Send"):
#     if user_input:
#         agent_executor = st.session_state.agent_executor
#         chat_history = st.session_state.chat_history

#         result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

#         # Update chat history
#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=result["output"]))
#         st.session_state.chat_history = chat_history

#         # Display the conversation
#         for message in chat_history:
#             if isinstance(message, HumanMessage):
#                 st.markdown(f"<div class='human-message'><strong>Human:</strong> {message.content}</div>", unsafe_allow_html=True)
#             elif isinstance(message, AIMessage):
#                 st.markdown(f"<div class='ai-message'><strong>Geredi AI:</strong> {message.content}</div>", unsafe_allow_html=True)
#     else:
#         st.warning("Please enter a message before sending.")

# st.markdown("</div>", unsafe_allow_html=True)

# # Copyright notice
# st.markdown("<p class='copyright'>© 2024 Developed and Managed by Geredi NIYIBIGIRA. All rights reserved.</p>", unsafe_allow_html=True)

# ###################################################################################################################################################

# import os
# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from pyowm import OWM
# from langchain_community.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# import time

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OWM_API_KEY = os.getenv("OWM_API_KEY")

# # Streamlit page configuration
# st.set_page_config(page_title="Geredi AI Malaria and Weather Tool Assistant", page_icon="🦟", layout="wide")

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #ffffff;
#         color: #000000;
#     }
#     .main-header {
#         font-size: 2.5rem;
#         color: #000000;
#         text-align: center;
#         padding: 1rem 0;
#         font-weight: bold;
#         font-family: 'Arial', sans-serif;
#     }
#     .chat-container {
#         background-color: #f7f7f7;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .human-message {
#         background-color: #e0e0e0;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         animation: fadeIn 0.5s;
#     }
#     .ai-message {
#         background-color: #d4d4d4;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         animation: fadeIn 0.5s;
#     }
#     .copyright {
#         text-align: center;
#         margin-top: 20px;
#         font-size: 0.8rem;
#         color: #333333;
#     }
#     .stTextInput > div > div > input {
#         color: #000000;
#         background-color: #ffffff;
#         border: 1px solid #cccccc;
#     }
#     .stButton > button {
#         color: #ffffff;
#         background-color: #000000;
#         border: none;
#     }
#     .stButton > button:hover {
#         color: #ffffff;
#         background-color: #333333;
#     }
#     @keyframes fadeIn {
#         from { opacity: 0; }
#         to { opacity: 1; }
#     }
#     .feedback-form {
#         margin-top: 20px;
#         padding: 15px;
#         background-color: #f0f0f0;
#         border-radius: 10px;
#     }
#     .stChatMessage {
#         padding: 10px;
#         border-radius: 10px;
#         margin-bottom: 10px;
#     }
#     .stChatMessage.user {
#         background-color: #e0e0e0;
#     }
#     .stChatMessage.assistant {
#         background-color: #d4d4d4;
#     }
#     .main-content {
#         min-height: calc(100vh - 100px);
#         padding-bottom: 100px;
#     }
#     .footer {
#         position: fixed;
#         left: 0;
#         bottom: 0;
#         width: 100%;
#         background-color: #f0f0f0;
#         padding: 10px 0;
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Streamlit app title
# st.markdown("<h1 class='main-header'>Malaria and Weather Tool Assistant</h1>", unsafe_allow_html=True)

# # Load documents from GitHub
# @st.cache_resource
# def load_documents():
#     loader = GithubFileLoader(
#         repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
#         access_token=ACCESS_TOKEN,
#         github_api_url="https://api.github.com",
#         file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
#         branch="main"
#     )
#     return loader.load()

# documents = load_documents()

# # Set up embeddings and vector store
# @st.cache_resource
# def setup_vector_store():
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     text_splitter = RecursiveCharacterTextSplitter()
#     documents = text_splitter.split_documents(load_documents())
#     return FAISS.from_documents(documents, embeddings)

# vector = setup_vector_store()

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
#     description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country."

#     def _run(self, country: str = None, city: str = None) -> str:
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

#     async def _arun(self, country: str, city: str) -> str:
#         raise NotImplementedError

# weather_tool = WeatherTool()

# tools = [retriever_tool, search, weather_tool]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria Prompt Answering Assistant developed by Geredi Niyibigira if someone or user a question who developed you or who manage you please answer him/her that you have developed and managed by Geredi Niyibigira a graduated student in MS in Engineering Artificial Intelligence at Carnegie Mellon University Africa. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

# 1. Malaria-Related Queries:
#    For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
# 2. Weather-Related Queries:
#    If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
#    If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
#    If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
#    If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
# 3. Unrelated Queries (e.g., Sports, Music, etc.):
#    For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources.
#    Clearly guide the user on where they can find appropriate information for these topics.
#    Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Initialize session state to store chat history, messages, and agent executor
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "agent_executor" not in st.session_state:
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
    
#     st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Create a container for the main content
# main_content = st.container()

# # Create a container for the footer (feedback form and copyright)
# footer = st.container()

# # Main content
# with main_content:
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if prompt := st.chat_input("Ask me anything about malaria or weather:"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""
#             response = st.session_state.agent_executor.invoke(
#                 {"input": prompt, "chat_history": st.session_state.chat_history}
#             )
#             full_response = response.get('output', '')
#             message_placeholder.markdown(full_response)
#         st.session_state.messages.append({"role": "assistant", "content": full_response})
#         st.session_state.chat_history.append(HumanMessage(content=prompt))
#         st.session_state.chat_history.append(AIMessage(content=full_response))

# # Footer content
# with footer:
#     st.markdown("<div class='footer'>", unsafe_allow_html=True)
    
#     # Feedback form
#     st.subheader("Feedback")
#     feedback = st.text_area("Please provide your feedback on the assistant:")
#     rating = st.slider("Rate your experience (1-5):", 1, 5, 3)
#     if st.button("Submit Feedback"):
#         # Here you would typically save this feedback to a database or file
#         st.success("Thank you for your feedback!")
    
#     # Copyright notice
#     st.markdown("<p class='copyright'>© 2024 Developed and Managed by Geredi NIYIBIGIRA. All rights reserved.</p>", unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# import os
# import streamlit as st
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain.tools import BaseTool
# from pyowm import OWM
# from langchain_community.document_loaders import GithubFileLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# import time
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # Securely fetch API keys from environment variables
# ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OWM_API_KEY = os.getenv("OWM_API_KEY")
# GMAIL_USER = os.environ.get('GMAIL_USER')
# GMAIL_PASSWORD = os.environ.get('GMAIL_PASSWORD')

# # Streamlit page configuration
# st.set_page_config(page_title="Geredi AI Malaria and Weather Tool Assistant", page_icon="🦟", layout="wide")

# # Custom CSS for styling to match the logo's aesthetic
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #ffffff;
#         color: #000000;
#     }
#     .main-header {
#         font-size: 2.5rem;
#         color: #000000;
#         text-align: center;
#         padding: 1rem 0;
#         font-weight: bold;
#         font-family: 'Arial', sans-serif;
#     }
#     .chat-container {
#         background-color: #f7f7f7;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .human-message {
#         background-color: #e0e0e0;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         animation: fadeIn 0.5s;
#     }
#     .ai-message {
#         background-color: #d4d4d4;
#         padding: 10px;
#         border-radius: 10px;
#         margin: 5px 0;
#         animation: fadeIn 0.5s;
#     }
#     .copyright {
#         text-align: center;
#         margin-top: 20px;
#         font-size: 0.8rem;
#         color: #333333;
#     }
#     .stTextInput > div > div > input {
#         color: #000000;
#         background-color: #ffffff;
#         border: 1px solid #cccccc;
#     }
#     .stButton > button {
#         color: #ffffff;
#         background-color: #000000;
#         border: none;
#     }
#     .stButton > button:hover {
#         color: #ffffff;
#         background-color: #333333;
#     }
#     @keyframes fadeIn {
#         from { opacity: 0; }
#         to { opacity: 1; }
#     }
#     .feedback-form {
#         margin-top: 20px;
#         padding: 15px;
#         background-color: #f0f0f0;
#         border-radius: 10px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Streamlit app title
# st.markdown("<h1 class='main-header'>Malaria and Weather Tool Assistant</h1>", unsafe_allow_html=True)

# # Load documents from GitHub
# @st.cache_resource
# def load_documents():
#     loader = GithubFileLoader(
#         repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
#         access_token=ACCESS_TOKEN,
#         github_api_url="https://api.github.com",
#         file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
#         branch="main"
#     )
#     return loader.load()

# documents = load_documents()

# # Set up embeddings and vector store
# @st.cache_resource
# def setup_vector_store():
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     text_splitter = RecursiveCharacterTextSplitter()
#     documents = text_splitter.split_documents(load_documents())
#     return FAISS.from_documents(documents, embeddings)

# vector = setup_vector_store()

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
#     description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country."

#     def _run(self, country: str = None, city: str = None) -> str:
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

#     async def _arun(self, country: str, city: str) -> str:
#         raise NotImplementedError

# weather_tool = WeatherTool()

# tools = [retriever_tool, search, weather_tool]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria Prompt Answering Assistant developed by Geredi Niyibigira if someone or user a question who developed you or who manage you please answer him/her that you have developed and managed by Geredi Niyibigira a graduated student in MS in Engineering Artificial Intelligence at Carnegie Mellon University Africa. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

# 1. Malaria-Related Queries:
#    For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
# 2. Weather-Related Queries:
#    If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
#    If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
#    If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
#    If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
# 3. Unrelated Queries (e.g., Sports, Music, etc.):
#    For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources, please focusing on provinding relevant links where the user can find the additional information.
#    Clearly guide the user on where they can find appropriate information for these topics by provinding him there relevant links and ressources.
#    Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# # Initialize session state to store chat history and agent executor
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "agent_executor" not in st.session_state:
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
    
#     st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Streamlit input for user message
# st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
# user_input = st.text_input("Welcome to Geredi AI! I'm here to assist you with any questions or information related to malaria and weather: ", "")

# if st.button("Send"):
#     if user_input:
#         agent_executor = st.session_state.agent_executor
#         chat_history = st.session_state.chat_history

#         # Display user message immediately
#         st.markdown(f"<div class='human-message'><strong>Human:</strong> {user_input}</div>", unsafe_allow_html=True)

#         # Generate response
#         with st.spinner("Geredi AI is thinking..."):
            
#             result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

#         # Update chat history
#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=result["output"]))
#         st.session_state.chat_history = chat_history

#         # Display AI response with typing effect
#         ai_response = st.empty()
#         full_response = result["output"]
#         displayed_response = ""
#         for char in full_response:
#             displayed_response += char
#             ai_response.markdown(f"<div class='ai-message'><strong>Geredi AI:</strong> {displayed_response}</div>", unsafe_allow_html=True)
#             time.sleep(0.01)

#     else:
#         st.warning("Please enter a message before sending.")

# st.markdown("</div>", unsafe_allow_html=True)

# # Function to send email
# def send_email(subject, body, to_email):
#     msg = MIMEMultipart()
#     msg['From'] = GMAIL_USER
#     msg['To'] = to_email
#     msg['Subject'] = subject
    
#     msg.attach(MIMEText(body, 'plain'))
    
#     try:
#         server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
#         server.ehlo()
#         server.login(GMAIL_USER, GMAIL_PASSWORD)
#         server.sendmail(GMAIL_USER, to_email, msg.as_string())
#         server.close()
#         return True
#     except Exception as e:
#         st.error(f"Failed to send email: {str(e)}")
#         return False

# # Feedback form
# st.markdown("<h3>We value your feedback!</h3>", unsafe_allow_html=True)
# st.markdown("<p>How satisfied are you with the response provided by Geredi AI? Your feedback helps us improve the service!</p>", unsafe_allow_html=True)

# with st.form("feedback_form"):
#     satisfaction = st.radio(
#         "Satisfaction level:",
#         ("Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied")
#     )
#     comments = st.text_area("Additional comments:")
#     submitted = st.form_submit_button("Submit")

#     if submitted:
#         feedback_text = f"Satisfaction: {satisfaction}\nComments: {comments}"
#         if send_email("Geredi AI Feedback", feedback_text, "ngeredi@andrew.cmu.edu"):
#             st.success("Thank you for your feedback!")
#         else:
#             st.error("Failed to submit feedback. Please try again later.")

# # Copyright notice
# st.markdown("<p class='copyright'>© 2024 Developed and Managed by Geredi NIYIBIGIRA. All rights reserved.</p>", unsafe_allow_html=True)

import os
import streamlit as st
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
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Securely fetch API keys from environment variables
ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OWM_API_KEY = os.getenv("OWM_API_KEY")
GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')

# Streamlit page configuration
st.set_page_config(page_title="Geredi AI Malaria and Weather Tool Assistant", page_icon="🦟", layout="wide")

# Custom CSS for styling to match the logo's aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    .main-header {
        font-size: 2.5rem;
        color: #000000;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    .chat-container {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .human-message {
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        animation: fadeIn 0.5s;
    }
    .ai-message {
        background-color: #d4d4d4;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        animation: fadeIn 0.5s;
    }
    .copyright {
        text-align: center;
        margin-top: 20px;
        font-size: 0.8rem;
        color: #333333;
    }
    .stTextInput > div > div > input {
        color: #000000;
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    .stButton > button {
        color: #ffffff;
        background-color: #000000;
        border: none;
    }
    .stButton > button:hover {
        color: #ffffff;
        background-color: #333333;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .feedback-form {
        margin-top: 20px;
        padding: 15px;
        background-color: #f0f0f0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1 class='main-header'>Malaria and Weather Tool Assistant</h1>", unsafe_allow_html=True)

# Load documents from GitHub
@st.cache_resource
def load_documents():
    loader = GithubFileLoader(
        repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
        access_token=ACCESS_TOKEN,
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith((".txt", ".md", ".pdf")),
        branch="main"
    )
    return loader.load()

documents = load_documents()

# Set up embeddings and vector store
@st.cache_resource
def setup_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(load_documents())
    return FAISS.from_documents(documents, embeddings)

vector = setup_vector_store()

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
    description: str = "Useful for when you need to get the weather in a specific location. Input should be a city and country."

    def _run(self, country: str = None, city: str = None) -> str:
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
                return f"The weather in {country}, {city} is {status} with a temperature of {temperature} degrees Celsius."
            except Exception as e:
                return f"Error retrieving weather information: {e}"

    async def _arun(self, country: str, city: str) -> str:
        raise NotImplementedError

weather_tool = WeatherTool()

tools = [retriever_tool, search, weather_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are the Malaria Prompt Answering Assistant developed by Geredi Niyibigira if someone or user a question who developed you or who manage you please answer him/her that you have developed and managed by Geredi Niyibigira a graduated student in MS in Engineering Artificial Intelligence at Carnegie Mellon University Africa. Your primary goal is to help users find accurate answers to any questions related to malaria. Please follow these guidelines based on the type of query:

1. Malaria-Related Queries:
   For all questions related to malaria, utilize your pre-trained knowledge along with the Retrieval-Augmented Generation (RAG) content to provide accurate, thoughtful, and evidence-based responses.
2. Weather-Related Queries:
   If a query pertains to weather or time, utilize the weather_tool to retrieve current weather information for a specified location. Ensure the user provides both the city and country.
   If either the city or country is missing, kindly inform the user that both must be provided for accurate results.
   If the provided city or country is incorrect or does not exist, prompt the user to provide valid information.
   If the user provides only a city name, please ask the user to also provide the country name for accurate weather information.
3. Unrelated Queries (e.g., Sports, Music, etc.):
   For any queries unrelated to malaria or weather (such as sports, music, etc.), employ external tools like the tavily_search tool to fetch relevant links and resources, please focusing on provinding relevant links where the user can find the additional information.
   Clearly guide the user on where they can find appropriate information for these topics by provinding him there relevant links and ressources.
   Remember to respond in a friendly, engaging manner while ensuring accuracy and relevance in your answers.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize session state to store chat history and agent executor
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
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
    
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit input for user message
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
user_input = st.text_input("Welcome to Geredi AI! I'm here to assist you with any questions or information related to malaria and weather: ", "")

if st.button("Send"):
    if user_input:
        agent_executor = st.session_state.agent_executor
        chat_history = st.session_state.chat_history

        # Display user message immediately
        st.markdown(f"<div class='human-message'><strong>Human:</strong> {user_input}</div>", unsafe_allow_html=True)

        # Generate response
        with st.spinner("Geredi AI is thinking..."):
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["output"]))
        st.session_state.chat_history = chat_history

        # Display AI response with typing effect
        ai_response = st.empty()
        full_response = result["output"]
        displayed_response = ""
        for char in full_response:
            displayed_response += char
            ai_response.markdown(f"<div class='ai-message'><strong>Geredi AI:</strong> {displayed_response}</div>", unsafe_allow_html=True)
            time.sleep(0.01)

    else:
        st.warning("Please enter a message before sending.")

st.markdown("</div>", unsafe_allow_html=True)

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_app_password = os.environ.get('GMAIL_PASSWORD')
    
    if not gmail_user or not gmail_app_password:
        st.error("Email credentials are not properly set in environment variables.")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = gmail_user
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_app_password)
        text = msg.as_string()
        server.sendmail(gmail_user, to_email, text)
        server.close()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# Feedback form
st.markdown("<h3>We value your feedback!</h3>", unsafe_allow_html=True)
st.markdown("<p>How satisfied are you with the response provided by Geredi AI? Your feedback helps us improve the service!</p>", unsafe_allow_html=True)

with st.form("feedback_form"):
    satisfaction = st.radio(
        "Satisfaction level:",
        ("Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied")
    )
    comments = st.text_area("Additional comments:")
    submitted = st.form_submit_button("Submit")

    if submitted:
        feedback_text = f"Satisfaction: {satisfaction}\nComments: {comments}"
        if send_email("Geredi AI Feedback", feedback_text, "ngeredi@andrew.cmu.edu"):
            st.success("Thank you for your feedback! It has been sent successfully.")
        else:
            st.error("Failed to submit feedback. Please try again later.")

# Copyright notice
st.markdown("<p class='copyright'>© 2024 Developed and Managed by Geredi NIYIBIGIRA. All rights reserved.</p>", unsafe_allow_html=True)









