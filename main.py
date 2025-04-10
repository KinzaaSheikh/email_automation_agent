import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is set
if not gemini_api_key:  
    raise ValueError("GEMINI_API_KEY is not set in the .env file")


# Configure the OpenAI-compatible client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create the model with the configured client
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Define a structured output model using Pydantic
class CapitalInfo(BaseModel):
    country: str
    capital: str
    population: int
    fun_fact: str

# Define a structured input model using Pydantic
structured_input = Agent(
    name="capital_info_agent",
    instructions="You are a helpful assistant that provides information about a country's capital, population, and fun fact.",
    output_type=CapitalInfo,
    model=model
)

# Create a coroutine to run the agent
coro = Runner.run_sync(
    structured_input,
    [{"role": "user", "content": "What is the capital of France?"}]
)

def main():
    print(coro.final_output)

if __name__ == "__main__":
    main()
