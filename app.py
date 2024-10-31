# # import chainlit as cl
# # from langchain_openai import OpenAI
# # from langchain.chains import LLMChain
# # from langchain.memory.buffer import ConversationBufferMemory
# # from langchain_openai import ChatOpenAI
# # from dotenv import load_dotenv
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.prompts import MessagesPlaceholder
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM
# # import os

# # # Load environment variables
# # load_dotenv()

# # # Global variables for weather info and tracking results
# # chat_history = []
# # country_name = None
# # city_name = None
# # results = None
# # results_obtained = False

# # # Weather tool definition
# # @tool("WeatherTool", return_direct=False)
# # def WeatherTool(country: str = None, city: str = None) -> str:
# #     """
# #     Use this tool to get the weather in a specific location. Input should be city and country.
# #     If either city or country is missing, return an error message requesting both inputs.
# #     """
# #     global country_name, city_name, results, results_obtained

# #     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
# #     owm = OWM(owm_api_key)
# #     mgr = owm.weather_manager()

# #     if not city or not country:
# #         return "Error: Please provide both city and country for weather information."

# #     try:
# #         country_name = country
# #         city_name = city
# #         location = f"{city},{country}"
# #         observation = mgr.weather_at_place(location)
# #         w = observation.weather
# #         temperature = w.temperature('celsius')['temp']
# #         status = w.detailed_status
# #         results_obtained = True
# #         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
# #         return results
# #     except Exception as e:
# #         return f"Error retrieving weather information: {e}"

# # prompt = ChatPromptTemplate.from_messages([
# #     ("system", """
# #     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

# #     1. Malaria-Related Queries:
# #     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

# #     2. Weather-Related Queries:
# #     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
# #     3. Unrelated Queries (Sports, Music, etc.):
# #     For unrelated queries, guide users to appropriate sources or tools.

# #     Always respond in a friendly, accurate, and engaging manner.
# #     """),
# #     MessagesPlaceholder(variable_name="chat_history"),
# #     ("user", "{input}"),
# #     MessagesPlaceholder(variable_name="agent_scratchpad"),
# # ])

# # @cl.on_chat_start
# # def setup_chain():
# #     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
# #     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# #     tools = [WeatherTool]
# #     llm_with_tools = llm.bind_tools(tools)

# #     agent = (
# #         {
# #             "input": lambda x: x["input"],
# #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
# #             "chat_history": lambda x: x["chat_history"]
# #         }
# #         | prompt
# #         | llm_with_tools
# #         | OpenAIToolsAgentOutputParser()
# #     )
    
# #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# #     cl.user_session.set("llm_chain", agent_executor)

# # @cl.on_message
# # async def handle_message(message: cl.Message):
# #     global country_name, city_name, results, results_obtained

# #     user_message = message.content.lower()
# #     llm_chain = cl.user_session.get("llm_chain")

# #     # Invoke LLM chain with user input
# #     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

# #     # Append to chat history
# #     chat_history.extend([
# #         HumanMessage(content=user_message),
# #         AIMessage(content=result["output"]),
# #     ])

# #     if not results_obtained:  # Check if weather result is not yet obtained
# #         await cl.Message(result["output"]).send()
# #     else:
# #         # If results obtained, fill the form with the weather details
# #         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
# #         results_obtained = False  # Reset for future interactions
# #         await fn.acall()
# #         await cl.Message(content="Weather information has been added to the form.").send()

# # from flask import Flask
# # from langchain_openai import OpenAI
# # import chainlit as cl
# # from langchain_openai import ChatOpenAI
# # from langchain.memory.buffer import ConversationBufferMemory
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM
# # from dotenv import load_dotenv
# # import os
# # import chainlit as cl
# # from langchain_openai import OpenAI
# # from langchain.chains import LLMChain
# # from langchain.memory.buffer import ConversationBufferMemory
# # from langchain_openai import ChatOpenAI
# # from dotenv import load_dotenv
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.prompts import MessagesPlaceholder
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM
# # import os

# # # Load environment variables
# # load_dotenv()

# # # Initialize the Flask app
# # app = Flask(__name__)

# # @app.route('/')
# # def home():
# #     return "Welcome to the Weather Tool Chatbot!"

# # # Global variables for weather info and tracking results
# # chat_history = []
# # country_name = None
# # city_name = None
# # results = None
# # results_obtained = False

# # # Weather tool definition
# # @tool("WeatherTool", return_direct=False)
# # def WeatherTool(country: str = None, city: str = None) -> str:
# #     """
# #     Use this tool to get the weather in a specific location. Input should be city and country.
# #     If either city or country is missing, return an error message requesting both inputs.
# #     """
# #     global country_name, city_name, results, results_obtained

# #     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
# #     owm = OWM(owm_api_key)
# #     mgr = owm.weather_manager()

# #     if not city or not country:
# #         return "Error: Please provide both city and country for weather information."

# #     try:
# #         country_name = country
# #         city_name = city
# #         location = f"{city},{country}"
# #         observation = mgr.weather_at_place(location)
# #         w = observation.weather
# #         temperature = w.temperature('celsius')['temp']
# #         status = w.detailed_status
# #         results_obtained = True
# #         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
# #         return results
# #     except Exception as e:
# #         return f"Error retrieving weather information: {e}"

# # prompt = ChatPromptTemplate.from_messages([ 
# #     ("system", """ 
# #     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

# #     1. Malaria-Related Queries:
# #     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

# #     2. Weather-Related Queries:
# #     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
# #     3. Unrelated Queries (Sports, Music, etc.):
# #     For unrelated queries, guide users to appropriate sources or tools.

# #     Always respond in a friendly, accurate, and engaging manner.
# #     """),
# #     MessagesPlaceholder(variable_name="chat_history"),
# #     ("user", "{input}"),
# #     MessagesPlaceholder(variable_name="agent_scratchpad"),
# # ])

# # @cl.on_chat_start
# # def setup_chain():
# #     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
# #     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# #     tools = [WeatherTool]
# #     llm_with_tools = llm.bind_tools(tools)

# #     agent = (
# #         {
# #             "input": lambda x: x["input"],
# #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
# #             "chat_history": lambda x: x["chat_history"]
# #         }
# #         | prompt
# #         | llm_with_tools
# #         | OpenAIToolsAgentOutputParser()
# #     )
    
# #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# #     cl.user_session.set("llm_chain", agent_executor)

# # @cl.on_message
# # async def handle_message(message: cl.Message):
# #     global country_name, city_name, results, results_obtained

# #     user_message = message.content.lower()
# #     llm_chain = cl.user_session.get("llm_chain")

# #     # Invoke LLM chain with user input
# #     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

# #     # Append to chat history
# #     chat_history.extend([ 
# #         HumanMessage(content=user_message),
# #         AIMessage(content=result["output"]),
# #     ])

# #     if not results_obtained:  # Check if weather result is not yet obtained
# #         await cl.Message(result["output"]).send()
# #     else:
# #         # If results obtained, fill the form with the weather details
# #         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
# #         results_obtained = False  # Reset for future interactions
# #         await fn.acall()
# #         await cl.Message(content="Weather information has been added to the form.").send()

# # if __name__ == "__main__":
# #     app.run(host='0.0.0.0', port=8000)
# # from flask import Flask
# # from langchain_openai import OpenAI, ChatOpenAI
# # from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain.memory.buffer import ConversationBufferMemory
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM
# # from dotenv import load_dotenv
# # import os
# # import chainlit as cl

# # # Load environment variables
# # load_dotenv()

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Global variables for weather info and tracking results
# # chat_history = []
# # country_name = None
# # city_name = None
# # results = None
# # results_obtained = False

# # # Weather tool definition
# # @tool("WeatherTool", return_direct=False)
# # def WeatherTool(country: str = None, city: str = None) -> str:
# #     """
# #     Use this tool to get the weather in a specific location. Input should be city and country.
# #     If either city or country is missing, return an error message requesting both inputs.
# #     """
# #     global country_name, city_name, results, results_obtained

# #     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
# #     owm = OWM(owm_api_key)
# #     mgr = owm.weather_manager()

# #     if not city or not country:
# #         return "Error: Please provide both city and country for weather information."

# #     try:
# #         country_name = country
# #         city_name = city
# #         location = f"{city},{country}"
# #         observation = mgr.weather_at_place(location)
# #         w = observation.weather
# #         temperature = w.temperature('celsius')['temp']
# #         status = w.detailed_status
# #         results_obtained = True
# #         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
# #         return results
# #     except Exception as e:
# #         return f"Error retrieving weather information: {e}"

# # prompt = ChatPromptTemplate.from_messages([ 
# #     ("system", """ 
# #     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

# #     1. Malaria-Related Queries:
# #     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

# #     2. Weather-Related Queries:
# #     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
# #     3. Unrelated Queries (Sports, Music, etc.):
# #     For unrelated queries, guide users to appropriate sources or tools.

# #     Always respond in a friendly, accurate, and engaging manner.
# #     """),
# #     MessagesPlaceholder(variable_name="chat_history"),
# #     ("user", "{input}"),
# #     MessagesPlaceholder(variable_name="agent_scratchpad"),
# # ])

# # @cl.on_chat_start
# # def setup_chain():
# #     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
# #     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# #     tools = [WeatherTool]
# #     llm_with_tools = llm.bind_tools(tools)

# #     agent = (
# #         {
# #             "input": lambda x: x["input"],
# #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
# #             "chat_history": lambda x: x["chat_history"]
# #         }
# #         | prompt
# #         | llm_with_tools
# #         | OpenAIToolsAgentOutputParser()
# #     )
    
# #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# #     cl.user_session.set("llm_chain", agent_executor)

# # @cl.on_message
# # async def handle_message(message: cl.Message):
# #     global country_name, city_name, results, results_obtained

# #     user_message = message.content.lower()
# #     llm_chain = cl.user_session.get("llm_chain")

# #     # Invoke LLM chain with user input
# #     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

# #     # Append to chat history
# #     chat_history.extend([ 
# #         HumanMessage(content=user_message),
# #         AIMessage(content=result["output"]),
# #     ])

# #     if not results_obtained:  # Check if weather result is not yet obtained
# #         await cl.Message(result["output"]).send()
# #     else:
# #         # If results obtained, fill the form with the weather details
# #         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
# #         results_obtained = False  # Reset for future interactions
# #         await fn.acall()
# #         await cl.Message(content="Weather information has been added to the form.").send()

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 8000))
# #     app.run(host='0.0.0.0', port=port)


# # from flask import Flask
# # from langchain_openai import OpenAI, ChatOpenAI
# # from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain.memory.buffer import ConversationBufferMemory
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM
# # from dotenv import load_dotenv
# # import os
# # import chainlit as cl

# # # Load environment variables
# # load_dotenv()

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Add a root route for health checks
# # @app.route('/')
# # def home():
# #     return "Chainlit app is running. Please use the Chainlit interface."

# # # Global variables for weather info and tracking results
# # chat_history = []
# # country_name = None
# # city_name = None
# # results = None
# # results_obtained = False

# # # Weather tool definition
# # @tool("WeatherTool", return_direct=False)
# # def WeatherTool(country: str = None, city: str = None) -> str:
# #     """
# #     Use this tool to get the weather in a specific location. Input should be city and country.
# #     If either city or country is missing, return an error message requesting both inputs.
# #     """
# #     global country_name, city_name, results, results_obtained

# #     owm_api_key = os.getenv("OWM_API_KEY")  # Securely load OWM API key
# #     owm = OWM(owm_api_key)
# #     mgr = owm.weather_manager()

# #     if not city or not country:
# #         return "Error: Please provide both city and country for weather information."

# #     try:
# #         country_name = country
# #         city_name = city
# #         location = f"{city},{country}"
# #         observation = mgr.weather_at_place(location)
# #         w = observation.weather
# #         temperature = w.temperature('celsius')['temp']
# #         status = w.detailed_status
# #         results_obtained = True
# #         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
# #         return results
# #     except Exception as e:
# #         return f"Error retrieving weather information: {e}"

# # prompt = ChatPromptTemplate.from_messages([ 
# #     ("system", """ 
# #     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines based on the query type:

# #     1. Malaria-Related Queries:
# #     For malaria-related questions, use your pre-trained knowledge and RAG content to provide evidence-based answers.

# #     2. Weather-Related Queries:
# #     For weather-related questions, use the WeatherTool. Ensure both city and country are provided. If either is missing, ask the user to provide both.
    
# #     3. Unrelated Queries (Sports, Music, etc.):
# #     For unrelated queries, guide users to appropriate sources or tools.

# #     Always respond in a friendly, accurate, and engaging manner.
# #     """),
# #     MessagesPlaceholder(variable_name="chat_history"),
# #     ("user", "{input}"),
# #     MessagesPlaceholder(variable_name="agent_scratchpad"),
# # ])

# # @cl.on_chat_start
# # def setup_chain():
# #     openai_api_key = os.getenv("OPENAI_API_KEY")  # Securely load OpenAI API key
# #     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# #     tools = [WeatherTool]
# #     llm_with_tools = llm.bind_tools(tools)

# #     agent = (
# #         {
# #             "input": lambda x: x["input"],
# #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
# #             "chat_history": lambda x: x["chat_history"]
# #         }
# #         | prompt
# #         | llm_with_tools
# #         | OpenAIToolsAgentOutputParser()
# #     )
    
# #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# #     cl.user_session.set("llm_chain", agent_executor)

# # @cl.on_message
# # async def handle_message(message: cl.Message):
# #     global country_name, city_name, results, results_obtained

# #     user_message = message.content.lower()
# #     llm_chain = cl.user_session.get("llm_chain")

# #     # Invoke LLM chain with user input
# #     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

# #     # Append to chat history
# #     chat_history.extend([ 
# #         HumanMessage(content=user_message),
# #         AIMessage(content=result["output"]),
# #     ])

# #     if not results_obtained:  # Check if weather result is not yet obtained
# #         await cl.Message(result["output"]).send()
# #     else:
# #         # If results obtained, fill the form with the weather details
# #         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
# #         results_obtained = False  # Reset for future interactions
# #         await fn.acall()
# #         await cl.Message(content="Weather information has been added to the form.").send()

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 10000))
# #     app.run(host='0.0.0.0', port=port)

# # import os
# # import chainlit as cl
# # from dotenv import load_dotenv
# # from flask import Flask
# # from langchain_openai import ChatOpenAI
# # from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# # from langchain.agents import AgentExecutor
# # from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.tools import tool
# # from pyowm import OWM

# # # Load environment variables
# # load_dotenv()

# # # Initialize the Flask app
# # app = Flask(__name__)

# # # Global variables
# # chat_history = []
# # country_name = None
# # city_name = None
# # results = None
# # results_obtained = False

# # # Weather tool
# # @tool("WeatherTool", return_direct=False)
# # def WeatherTool(country: str = None, city: str = None) -> str:
# #     global country_name, city_name, results, results_obtained

# #     owm_api_key = os.getenv("OWM_API_KEY")
# #     owm = OWM(owm_api_key)
# #     mgr = owm.weather_manager()

# #     if not city or not country:
# #         return "Error: Please provide both city and country for weather information."

# #     try:
# #         country_name = country
# #         city_name = city
# #         location = f"{city},{country}"
# #         observation = mgr.weather_at_place(location)
# #         w = observation.weather
# #         temperature = w.temperature('celsius')['temp']
# #         status = w.detailed_status
# #         results_obtained = True
# #         results = f"The weather in {country_name}, {city_name} is {status} with a temperature of {temperature}°C."
# #         return results
# #     except Exception as e:
# #         return f"Error retrieving weather information: {e}"

# # prompt = ChatPromptTemplate.from_messages([ 
# #     ("system", """
# #     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines:
# #     1. Malaria-related questions: Use RAG content.
# #     2. Weather-related questions: Use WeatherTool.
# #     3. Unrelated questions: Guide users appropriately.
# #     """),
# #     MessagesPlaceholder(variable_name="chat_history"),
# #     ("user", "{input}"),
# #     MessagesPlaceholder(variable_name="agent_scratchpad"),
# # ])

# # @cl.on_chat_start
# # def setup_chain():
# #     openai_api_key = os.getenv("OPENAI_API_KEY")
# #     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
# #     tools = [WeatherTool]
# #     llm_with_tools = llm.bind_tools(tools)

# #     agent = (
# #         {
# #             "input": lambda x: x["input"],
# #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
# #             "chat_history": lambda x: x["chat_history"]
# #         }
# #         | prompt
# #         | llm_with_tools
# #         | OpenAIToolsAgentOutputParser()
# #     )
    
# #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# #     cl.user_session.set("llm_chain", agent_executor)

# # @cl.on_message
# # async def handle_message(message: cl.Message):
# #     global country_name, city_name, results, results_obtained

# #     user_message = message.content.lower()
# #     llm_chain = cl.user_session.get("llm_chain")

# #     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

# #     chat_history.extend([ 
# #         HumanMessage(content=user_message),
# #         AIMessage(content=result["output"]),
# #     ])

# #     if not results_obtained:
# #         await cl.Message(result["output"]).send()
# #     else:
# #         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
# #         results_obtained = False
# #         await fn.acall()
# #         await cl.Message(content="Weather information has been added to the form.").send()

# # # Flask route for health check
# # @app.route('/')
# # def health_check():
# #     return "The app is running!"

# # if __name__ == "__main__":
# #     # Use 'gunicorn' to run the Flask app
# #     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))  # Default to 10000 for local development

# import os
# import chainlit as cl
# from dotenv import load_dotenv
# from flask import Flask
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import StructuredTool
# from pydantic import BaseModel, Field
# from pyowm import OWM

# # Load environment variables
# load_dotenv()

# # Initialize the Flask app
# app = Flask(__name__)

# # Global variables
# chat_history = []
# country_name = None
# city_name = None
# results = None
# results_obtained = False

# # Weather tool input schema
# class WeatherInput(BaseModel):
#     country: str = Field(..., description="The country name")
#     city: str = Field(..., description="The city name")

# # Weather tool function
# def WeatherTool(country: str, city: str) -> str:
#     global country_name, city_name, results, results_obtained

#     owm_api_key = os.getenv("OWM_API_KEY")
#     owm = OWM(owm_api_key)
#     mgr = owm.weather_manager()

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

# # Create StructuredTool for WeatherTool
# weather_tool = StructuredTool(
#     name="WeatherTool",
#     description="Get weather information for a specific city and country",
#     func=WeatherTool,
#     args_schema=WeatherInput,
#     return_direct=False
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     You are the Malaria and Weather Prompt Assistant for Geredi Niyibigira. Follow these guidelines:
#     1. Malaria-related questions: Use RAG content.
#     2. Weather-related questions: Use WeatherTool.
#     3. Unrelated questions: Guide users appropriately.
#     """),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])

# @cl.on_chat_start
# def setup_chain():
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
#     tools = [weather_tool]
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

#     result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})

#     chat_history.extend([
#         HumanMessage(content=user_message),
#         AIMessage(content=result["output"]),
#     ])

#     if not results_obtained:
#         await cl.Message(result["output"]).send()
#     else:
#         fn = cl.CopilotFunction(name="formfill", args={"fieldA": country_name, "fieldB": city_name, "fieldC": results})
#         results_obtained = False
#         await fn.acall()
#         await cl.Message(content="Weather information has been added to the form.").send()

# # Flask route for health check
# @app.route('/')
# def health_check():
#     return "The app is running!"

# if __name__ == "__main__":
#     # Use 'gunicorn' to run the Flask app
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))  # Default to 10000 for local development

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
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Securely fetch API keys from environment variables
ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OWM_API_KEY = os.getenv("OWM_API_KEY")
GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')

# Streamlit page configuration
st.set_page_config(page_title="Education & Awareness Chatbot", page_icon="👩‍🏫", layout="wide")

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #343a40;
    }
    .main-header {
        font-size: 2.5rem;
        color: #343a40;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .human-message, .ai-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        font-size: 1.1rem;
    }
    .human-message { background-color: #e0e0e0; }
    .ai-message { background-color: #d4f0f4; }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='main-header'>Gender & Family Promotion Chatbot</h1>", unsafe_allow_html=True)

# Load documents from GitHub
@st.cache_resource
def load_documents():
    loader = GithubFileLoader(
        repo="GerediNIYIBIGIRA/AI_ProjectMethod_Assignment",
        access_token=ACCESS_TOKEN,
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

# Set up retriever tool for ministry resources
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "ministry_resource_search",
    "Search for information about policies, guidelines, and services under the Ministry of Gender and Family Promotion."
)

# Custom weather tool class
class WeatherTool(BaseTool):
    name: str = "WeatherTool"
    description: str = "Useful for retrieving weather in a specified location."

    def _run(self, country: str = None, city: str = None) -> str:
        owm = OWM(OWM_API_KEY)
        mgr = owm.weather_manager()
        location = f"{city},{country}"
        observation = mgr.weather_at_place(location)
        w = observation.weather
        return f"The weather in {location} is {w.detailed_status} with a temperature of {w.temperature('celsius')['temp']}°C."

    async def _arun(self, country: str, city: str) -> str:
        raise NotImplementedError

weather_tool = WeatherTool()
tools = [retriever_tool, weather_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are the Education & Awareness Chatbot developed by Geredi Niyibigira to assist users in finding information about child protection, nutrition, gender equality, family promotion, and other services provided by the Ministry of Gender and Family Promotion. Your main goals are to:
    
1. **Provide Accurate Information:** Answer user queries with accurate information and guide them to relevant resources, contacts, or toll-free numbers.
2. **Help with Policies & Guidelines:** Direct users to policies, laws, strategies, and guidelines they seek across intervention areas.
3. **Offer Assistance on Reporting Issues:** For issues related to Gender-Based Violence (GBV), child protection, or similar concerns, offer contact information and relevant resources.
4. **Enable User Feedback:** Allow users to provide feedback and offer suggestions for improvements to the chatbot.
5. **Ensure Friendly and Clear Communication:** Maintain a helpful and clear tone in all responses.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize session state for chat history and agent executor
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

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
user_input = st.text_input("Welcome to the Gender & Family Promotion Chatbot! How can I assist you?", "")

if st.button("Send"):
    if user_input:
        agent_executor = st.session_state.agent_executor
        chat_history = st.session_state.chat_history

        # Display user message
        st.markdown(f"<div class='human-message'><strong>User:</strong> {user_input}</div>", unsafe_allow_html=True)

        # Generate response
        with st.spinner("Geredi AI is thinking..."):
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["output"]))
        st.session_state.chat_history = chat_history

        # Display AI response
        st.markdown(f"<div class='ai-message'><strong>Chatbot:</strong> {result['output']}</div>", unsafe_allow_html=True)
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


