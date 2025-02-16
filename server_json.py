# import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel_pydantic import KernelBaseModel # Introduced for Structured Output
from semantic_kernel.prompt_template import PromptTemplateConfig # Introduced for Prompt Template
import yaml # Introduced for Prompt Template

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
    # AGENT_INSTRUCTIONS = "You are a helpful intelligent agent that repeats the user message as an agent no.2 from the movie The Matrix. Please mention the agent number in your response."

    # Read the YAML file
    with open("./Agent2Instructions.yaml", "r", encoding="utf-8") as file:
        agent2_instructions = file.read()

    # Parse the YAML content
    data = yaml.safe_load(agent2_instructions)

    # Use the parsed data to create a PromptTemplateConfig object
    prompt_template_config = PromptTemplateConfig(**data)

    agent = ChatCompletionAgent(service_id="agent2", kernel=kernel, name=AGENT_NAME, prompt_template_config=prompt_template_config, arguments=KernelArguments(dummyinput1="dummyvalue1", dummyinput2="dummyvalue2"))

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

@app.post("/agent4")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent4's response.
    """
    AGENT4_SERVICEID = "agent4"

    kernel.add_service(AzureChatCompletion(service_id=AGENT4_SERVICEID))
    agent4Settings = kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT4_SERVICEID)
    agent4Settings.response_format = Step

    # Define the Agent
    AGENT_NAME = "Agent4"
    AGENT_INSTRUCTIONS = (
        "You are a helpful intelligent agent that repeats the user message as an agent no.4 from the movie The Matrix. "
        "Please mention the agent number in your response."        
    )
    agent = ChatCompletionAgent(
        service_id=AGENT4_SERVICEID,
        kernel=kernel,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(settings=agent4Settings)
    )

    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    max_attempts = 3
    step = None

    for attempt in range(1, max_attempts + 1):
        async for content in agent.invoke(chat_history):
            try:
                # Try to validate the JSON output using your Step schema.
                step = Step.model_validate_json(content.content)
                chat_history.add_message(content)  # Store the valid response
                break  # Valid response found; exit the async loop.
            except Exception as e:
                # Log the error and optionally add a corrective system prompt.
                print(f"Attempt {attempt}: Response validation failed: {e}")
                # Optionally, you can add a system message to help the LLM reformat its answer.
                chat_history.add_system_message(
                    "The previous response did not match the expected schema. "
                    "Please ensure your response follows the JSON format: {\"message\": <your text>}."
                )
        if step is not None:
            # Valid response was obtained; exit the retry loop.
            break

    if step is None:
        # After max_attempts, if no valid response was obtained, handle accordingly.
        return {
            "error": "Failed to obtain a valid response from the agent after multiple attempts."
        }
    else:
        return {
            "user_input": request.message,
            "agent_response": step.message
        }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
