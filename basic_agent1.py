import datetime
import requests
from zoneinfo import ZoneInfo
from abc import ABC, abstractmethod
import requests
import json
import ast
import operator
# from openai import OpenAI
import os
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        pass


    @abstractmethod
    def description(self) -> str:
        pass


    @abstractmethod
    def use(self, *args, **kwargs):
        pass

class TimeTool(Tool):
    def name(self):
        return "Time Tool"


    def description(self):
        return "Provides the current time for a given city's timezone like Asia/Kolkata, America/New_York etc. If no timezone is provided, it returns the local time."


    def use(self, *args, **kwargs):
        format = "%Y-%m-%d %H:%M:%S %Z%z"
        current_time = datetime.datetime.now()
        input_timezone = args[0]
        if input_timezone:
            print("TimeZone", input_timezone)
            current_time =  current_time.astimezone(ZoneInfo(input_timezone))
        return f"The current time is {current_time}."


class WeatherTool(Tool):
    def name(self):
        return "Weather Tool"


    def description(self):
        return "Provides weather information for a given location"


    def use(self, *args, **kwargs):
        location = args[0].split("weather in ")[-1]
        weather_api_key = os.environ["OPEN_WEATHER_KEY"]
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"The weather in {location} is currently {description} with a temperature of {temp}°C."
        else:
            return f"Sorry, I couldn't find weather information for {location}."

class WebSearch(Tool):
    def name(self):
        return "Web search tool"
    

    def description(self):
        return "Searches the web for results for a given query. Uses the Tavily API based on the input query. Perform a web search using the Tavily API based on the input query. Parameters: query (str): The search query. Example: 'latest news on AI' Returns: str: A concise answer to the query or an error message."
    

    def use(self, *args, **kwargs):
        query = args[0]
        try:
            # Initialize Tavily client with API key from environment
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            
            # Get search results (simplified API call)
            response = client.search(query=query, include_answer=True)
            
            # Format results with error handling for missing fields
            if not response.get('answer'):
                return f"No results found for the query: {query}."
                
            return f"Search results for '{query}':\n{response['answer']}"
        
        except Exception as e:
            return f"Search failed: {str(e)}"

class CalculatorTool(Tool):
    def name(self):
        return "Calculator Tool"

    def description(self):
        return (
            "Performs numeric operations on two numbers. Provide a JSON string with 'num1', 'num2', and 'operation'. "
            "Supported operations: add, subtract, multiply, divide, floor_divide, modulus, power, "
            "lt (less than), le (less equal), eq (equal), ne (not equal), ge (greater equal), gt (greater than)."
        )

    def use(self, *args, **kwargs):
        if not args:
            return "Error: No input provided. Please provide a JSON string with 'num1', 'num2', and 'operation'."
        
        input_str = args[0]
        try:
            # Clean and parse input
            input_str_clean = input_str.replace("'", "\"").strip().strip("\"")
            input_dict = json.loads(input_str_clean)
            
            num1 = input_dict['num1']
            num2 = input_dict['num2']
            operation = input_dict['operation']
        except json.JSONDecodeError as e:
            return f"Invalid JSON format: {str(e)}"
        except KeyError as e:
            return f"Missing required field: {e}"
        except Exception as e:
            return f"Input error: {str(e)}"

        # Supported operations
        operations = {
            'add': operator.add,
            'subtract': operator.sub,
            'multiply': operator.mul,
            'divide': operator.truediv,
            'floor_divide': operator.floordiv,
            'modulus': operator.mod,
            'power': operator.pow,
            'lt': operator.lt,
            'le': operator.le,
            'eq': operator.eq,
            'ne': operator.ne,
            'ge': operator.ge,
            'gt': operator.gt
        }

        if operation not in operations:
            return f"Unsupported operation. Valid operations: {', '.join(operations.keys())}"

        try:
            result = operations[operation](num1, num2)
            return f"The result of {operation} on {num1} and {num2} is {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

class ReverserTool(Tool):
    def name(self):
        return "Reverser Tool"

    def description(self):
        return "Reverses the given input string"

    def use(self, *args, **kwargs):
        if not args:
            return "Error: No input provided. Please provide a string to reverse."
            
        input_string = args[0]
        
        if not isinstance(input_string, str):
            return "Error: Input must be a string"
            
        reversed_string = input_string[::-1]
        return f"The reversed string is: {reversed_string}"

class Agent:
    def __init__(self):
        self.tools = []
        self.memory = []
        self.max_memory = 10
        self.client=Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        # self.client=OpenAI(
        #     base_url='http://localhost:11434/v1',  # Ollama's API endpoint
        #     api_key='ollama',  # Required but unused for Ollama
        # )


    def add_tool(self, tool: Tool):
        self.tools.append(tool)


    def json_parser(self, input_string):
        """
        Parses a string into a Python dictionary and validates it.

        Parameters:
        input_string (str): A string representation of a dictionary.

        Returns:
        dict: The parsed dictionary.

        Raises:
        ValueError: If the input string cannot be parsed into a valid dictionary.
        """
        try:
            # Safely evaluate the input string into a Python object
            python_dict = ast.literal_eval(input_string)
            
            # Convert the Python object to a JSON string
            json_string = json.dumps(python_dict)
            
            # Parse the JSON string into a Python dictionary
            json_dict = json.loads(json_string)
            
            # Validate that the result is a dictionary
            if isinstance(json_dict, dict):
                return json_dict
            else:
                raise ValueError("Parsed object is not a dictionary")
        
        except (ValueError, SyntaxError, json.JSONDecodeError) as e:
            # Handle errors during parsing or validation
            raise ValueError(f"Invalid JSON response: {e}")


    def process_input(self, user_input):
        self.memory.append(f"User: {user_input}")
        self.memory = self.memory[-self.max_memory:]

        context = "\n".join(self.memory)
        tool_descriptions = "\n".join([f"- {tool.name()}: {tool.description()}" for tool in self.tools])
        response_format = {"action":"", "args":""}

        prompt = f"""Context:
        {context}

        Available tools:
        {tool_descriptions}

        Based on the user's input and context, decide if you should use a tool or respond directly.
        Sometimes you might have to use multiple tools to solve user's input. You have to do that in a loop.
        If you identify a action, respond with the tool name and the arguments for the tool.
        If you decide to respond directly to the user then make the action "respond_to_user" with args as your response in the following format.

        Response Format:
        Do not output anything apart from the following response format.

        {response_format}

        """

        response = self.query_llm(prompt)
        self.memory.append(f"Agent: {response}")
        response_dict = self.json_parser(response)

        # Check if any tool can handle the input
        for tool in self.tools:
            if tool.name().lower() == response_dict["action"].lower():
                return tool.use(response_dict["args"])

        return response_dict

    def query_llm(self, prompt):
        model_to_use = "llama-3.3-70b-versatile"

        # Create the completion request
        completion = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # Extract and return the response content
        final_response = completion.choices[0].message.content.strip()
        print("LLM Response:", final_response)
        return final_response
        

    def run(self):
      print("LLM Agent: Hello! How can I assist you today?")
      user_input = input("You: ")

      while True:
        if user_input.lower() in ["exit", "bye", "close"]:
          print("See you later!")
          break

        response = self.process_input(user_input)
        if isinstance(response, dict) and response["action"] == "respond_to_user":
          print("Reponse from Agent: ", response["args"])
          user_input = input("You: ")
        else:
          user_input = response

def main():
    agent = Agent()

    # Add tools to the agent
    agent.add_tool(TimeTool())
    agent.add_tool(WeatherTool())
    agent.add_tool(WebSearch())
    agent.add_tool(CalculatorTool())
    agent.add_tool(ReverserTool())
    agent.run()

if __name__ == "__main__":
    main()