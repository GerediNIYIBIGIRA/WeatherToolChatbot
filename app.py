# import chainlit as cl
# from langchain_openai import OpenAI
# from langchain.chains import LLMChain
# from langchain.memory.buffer import ConversationBufferMemory
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from pyowm import OWM
# import os

# # Load environment variables
# load_dotenv()

# # Global variables for weather info and tracking results
# chat_history = []
# country_name = None
# city_name = None
# results = None
# results_obtained = False

# # Weather tool definition
# @tool("WeatherTool", return_direct=False)
# def WeatherTool(country: str = None, city: str = None) -> str:
#     """
#     Use this tool to get the weather in a specific location. Input should be city and country.
#     If either city or country is missing, return an error message requesting both inputs.
#     """
#     global country_name, city_name, results, results_obtained

#     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
#     owm = OWM(owm_api_key)
#     mgr = owm.weather_manager()

#     if not city or not country:
#         return "Error: Please provide both city and country for weather information."

#     try:
#         country_name = country
#         city_name = city
#         location = f"{city},{country}"
#         observation = mgr.weather_at_place(location)
#         w = observation.weather
#         temperature = w.temperature('celsius')['temp']
#         status = w.detailed_status
#         results_obtained = True
#         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
#         return results
#     except Exception as e:
#         return f"Error retrieving weather information: {e}"

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

#     1. Malaria-Related Queries:
#     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

#     2. Weather-Related Queries:
#     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
#     3. Unrelated Queries (Sports, Music, etc.):
#     For unrelated queries, guide users to appropriate sources or tools.

#     Always respond in a friendly, accurate, and engaging manner.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# @cl.on_chat_start
# def setup_chain():
#     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
#     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
#     tools = [WeatherTool]
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
#     cl.user_session.set("llm_chain", agent_executor)

# @cl.on_message
# async def handle_message(message: cl.Message):
#     global country_name, city_name, results, results_obtained

#     user_message = message.content.lower()
#     llm_chain = cl.user_session.get("llm_chain")

#     # Invoke LLM chain with user input
#     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

#     # Append to chat history
#     chat_history.extend([
#         HumanMessage(content=user_message),
#         AIMessage(content=result["output"]),
#     ])

#     if not results_obtained:  # Check if weather result is not yet obtained
#         await cl.Message(result["output"]).send()
#     else:
#         # If results obtained, fill the form with the weather details
#         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
#         results_obtained = False  # Reset for future interactions
#         await fn.acall()
#         await cl.Message(content="Weather information has been added to the form.").send()

# from flask import Flask
# from langchain_openai import OpenAI
# import chainlit as cl
# from langchain_openai import ChatOpenAI
# from langchain.memory.buffer import ConversationBufferMemory
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from pyowm import OWM
# from dotenv import load_dotenv
# import os
# import chainlit as cl
# from langchain_openai import OpenAI
# from langchain.chains import LLMChain
# from langchain.memory.buffer import ConversationBufferMemory
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from pyowm import OWM
# import os

# # Load environment variables
# load_dotenv()

# # Initialize the Flask app
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Welcome to the Weather Tool Chatbot!"

# # Global variables for weather info and tracking results
# chat_history = []
# country_name = None
# city_name = None
# results = None
# results_obtained = False

# # Weather tool definition
# @tool("WeatherTool", return_direct=False)
# def WeatherTool(country: str = None, city: str = None) -> str:
#     """
#     Use this tool to get the weather in a specific location. Input should be city and country.
#     If either city or country is missing, return an error message requesting both inputs.
#     """
#     global country_name, city_name, results, results_obtained

#     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
#     owm = OWM(owm_api_key)
#     mgr = owm.weather_manager()

#     if not city or not country:
#         return "Error: Please provide both city and country for weather information."

#     try:
#         country_name = country
#         city_name = city
#         location = f"{city},{country}"
#         observation = mgr.weather_at_place(location)
#         w = observation.weather
#         temperature = w.temperature('celsius')['temp']
#         status = w.detailed_status
#         results_obtained = True
#         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
#         return results
#     except Exception as e:
#         return f"Error retrieving weather information: {e}"

# prompt = ChatPromptTemplate.from_messages([ 
#     ("system", """ 
#     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

#     1. Malaria-Related Queries:
#     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

#     2. Weather-Related Queries:
#     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
#     3. Unrelated Queries (Sports, Music, etc.):
#     For unrelated queries, guide users to appropriate sources or tools.

#     Always respond in a friendly, accurate, and engaging manner.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# @cl.on_chat_start
# def setup_chain():
#     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
#     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
#     tools = [WeatherTool]
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
#     cl.user_session.set("llm_chain", agent_executor)

# @cl.on_message
# async def handle_message(message: cl.Message):
#     global country_name, city_name, results, results_obtained

#     user_message = message.content.lower()
#     llm_chain = cl.user_session.get("llm_chain")

#     # Invoke LLM chain with user input
#     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

#     # Append to chat history
#     chat_history.extend([ 
#         HumanMessage(content=user_message),
#         AIMessage(content=result["output"]),
#     ])

#     if not results_obtained:  # Check if weather result is not yet obtained
#         await cl.Message(result["output"]).send()
#     else:
#         # If results obtained, fill the form with the weather details
#         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
#         results_obtained = False  # Reset for future interactions
#         await fn.acall()
#         await cl.Message(content="Weather information has been added to the form.").send()

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000)
from flask import Flask
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory.buffer import ConversationBufferMemory
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pyowm import OWM
from dotenv import load_dotenv
import os
import chainlit as cl

# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Global variables for weather info and tracking results
chat_history = []
country_name = None
city_name = None
results = None
results_obtained = False

# Weather tool definition
@tool("WeatherTool", return_direct=False)
def WeatherTool(country: str = None, city: str = None) -> str:
    """
    Use this tool to get the weather in a specific location. Input should be city and country.
    If either city or country is missing, return an error message requesting both inputs.
    """
    global country_name, city_name, results, results_obtained

    owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
    owm = OWM(owm_api_key)
    mgr = owm.weather_manager()

    if not city or not country:
        return "Error: Please provide both city and country for weather information."

    try:
        country_name = country
        city_name = city
        location = f"{city},{country}"
        observation = mgr.weather_at_place(location)
        w = observation.weather
        temperature = w.temperature('celsius')['temp']
        status = w.detailed_status
        results_obtained = True
        results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
        return results
    except Exception as e:
        return f"Error retrieving weather information: {e}"

prompt = ChatPromptTemplate.from_messages([ 
    ("system", """ 
    You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

    1. Malaria-Related Queries:
    For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

    2. Weather-Related Queries:
    For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
    3. Unrelated Queries (Sports, Music, etc.):
    For unrelated queries, guide users to appropriate sources or tools.

    Always respond in a friendly, accurate, and engaging manner.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@cl.on_chat_start
def setup_chain():
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    tools = [WeatherTool]
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
    cl.user_session.set("llm_chain", agent_executor)

@cl.on_message
async def handle_message(message: cl.Message):
    global country_name, city_name, results, results_obtained

    user_message = message.content.lower()
    llm_chain = cl.user_session.get("llm_chain")

    # Invoke LLM chain with user input
    result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

    # Append to chat history
    chat_history.extend([ 
        HumanMessage(content=user_message),
        AIMessage(content=result["output"]),
    ])

    if not results_obtained:  # Check if weather result is not yet obtained
        await cl.Message(result["output"]).send()
    else:
        # If results obtained, fill the form with the weather details
        fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
        results_obtained = False  # Reset for future interactions
        await fn.acall()
        await cl.Message(content="Weather information has been added to the form.").send()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)