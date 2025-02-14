# import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel_pydantic import KernelBaseModel # Introduced for Structured Output

# FastAPI App
app = FastAPI()

# Initialize Kernel
kernel = Kernel()


# Pydantic Model for Request Body
class ChatRequest(BaseModel):
    message: str

# For Structured Output
class Step(KernelBaseModel):
    message: str


@app.post("/agent1")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent1's response.
    """
    AGENT1_SERVICEID = "agent1"

    kernel.add_service(AzureChatCompletion(service_id=AGENT1_SERVICEID))
    agent1Settings = kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT1_SERVICEID)
    agent1Settings.response_format=Step

    # Define the Agent
    AGENT_NAME = "Agent1"
    AGENT_INSTRUCTIONS = "You are a helpful intelligent agent that repeats the user message as an agent no.1 from the movie The Matrix. Please mention the agent number in your response."
    agent = ChatCompletionAgent(service_id=AGENT1_SERVICEID, kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS, arguments=KernelArguments(settings=agent1Settings))

    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    response_text = ""
    async for content in agent.invoke(chat_history):
        step = Step.model_validate_json(content.content)
        chat_history.add_message(content) # Store the last response
        response_text = step.message  

    return {"user_input": request.message, "agent_response": response_text}


@app.post("/agent2")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent2's response.
    """
    kernel.add_service(AzureChatCompletion(service_id="agent2"))

    # Define the Agent
    AGENT_NAME = "Agent2"
    AGENT_INSTRUCTIONS = "You are a helpful intelligent agent that repeats the user message as an agent no.2 from the movie The Matrix. Please mention the agent number in your response."
    agent = ChatCompletionAgent(service_id="agent2", kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS)

    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    response_text = ""
    async for content in agent.invoke(chat_history):
        chat_history.add_message(content)
        response_text = content.content  # Store the last response

    return {"user_input": request.message, "agent_response": response_text}


@app.post("/agent3")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent3's response.
    """
    kernel.add_service(AzureChatCompletion(service_id="agent3"))

    # Define the Agent
    AGENT_NAME = "Agent3"
    AGENT_INSTRUCTIONS = "You are a helpful intelligent agent that repeats the user message as an agent no.3 from the movie The Matrix. Please mention the agent number in your response."
    agent = ChatCompletionAgent(service_id="agent3", kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS)

    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    response_text = ""
    async for content in agent.invoke(chat_history):
        chat_history.add_message(content)
        response_text = content.content  # Store the last response

    return {"user_input": request.message, "agent_response": response_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
