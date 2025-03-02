# import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.kernel_pydantic import KernelBaseModel # Introduced for Structured Output
from semantic_kernel.prompt_template import PromptTemplateConfig # Introduced for Prompt Template
import yaml # Introduced for Prompt Template

# Observability related imports
import logging
from azure.monitor.opentelemetry.exporter import (
    AzureMonitorLogExporter,
    AzureMonitorMetricExporter,
    AzureMonitorTraceExporter,
)

from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import DropAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import set_tracer_provider
# End of observability related imports

# Telemetry setup code goes here

# Replace the connection string with your Application Insights connection string
connection_string = "<App-Insights connection string here>"

# Create a resource to represent the service/sample
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "single_agent_server"})


def set_up_logging():
    exporter = AzureMonitorLogExporter(connection_string=connection_string)

    # Create and set a global logger provider for the application.
    logger_provider = LoggerProvider(resource=resource)
    # Log processors are initialized with an exporter which is responsible
    # for sending the telemetry data to a particular backend.
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    # Sets the global default logger provider
    set_logger_provider(logger_provider)

    # Create a logging handler to write logging records, in OTLP format, to the exporter.
    handler = LoggingHandler()
    # Add filters to the handler to only process records from semantic_kernel.
    handler.addFilter(logging.Filter("semantic_kernel"))
    # Attach the handler to the root logger. `getLogger()` with no arguments returns the root logger.
    # Events from all child loggers will be processed by this handler.
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def set_up_tracing():
    exporter = AzureMonitorTraceExporter(connection_string=connection_string)

    # Initialize a trace provider for the application. This is a factory for creating tracers.
    tracer_provider = TracerProvider(resource=resource)
    # Span processors are initialized with an exporter which is responsible
    # for sending the telemetry data to a particular backend.
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    # Sets the global default tracer provider
    set_tracer_provider(tracer_provider)


def set_up_metrics():
    exporter = AzureMonitorMetricExporter(connection_string=connection_string)

    # Initialize a metric provider for the application. This is a factory for creating meters.
    meter_provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(exporter, export_interval_millis=5000)],
        resource=resource,
        views=[
            # Dropping all instrument names except for those starting with "semantic_kernel"
            View(instrument_name="*", aggregation=DropAggregation()),
            View(instrument_name="semantic_kernel*"),
        ],
    )
    # Sets the global default meter provider
    set_meter_provider(meter_provider)


# This must be done before any other telemetry calls
set_up_logging()
set_up_tracing()
set_up_metrics()

# End of telemetry setup code


# FastAPI App
app = FastAPI()

# Initialize Kernel
kernel = Kernel()

from semantic_kernel.filters.prompts.prompt_render_context import PromptRenderContext
from semantic_kernel.filters.filter_types import FilterTypes
import re

@kernel.filter(FilterTypes.PROMPT_RENDERING)
async def credit_card_masking_filter(context: PromptRenderContext, next):
    """
    Filter to mask credit card numbers in the prompt.
    Assumes credit card numbers are exactly 16 digits long.
    The first 12 digits are replaced with '*' and the last 4 digits are retained.
    """
    # Continue to the next filter or rendering step.
    print("From the filter: Credit card masking filter")
    await next(context)
    
    # Regex pattern to match a 16-digit credit card number (captures first 12 and last 4 digits).
    pattern = re.compile(r'\b(\d{12})(\d{4})\b')
    
    # If there is a rendered prompt, substitute any found credit card numbers.
    if context.rendered_prompt:
        context.rendered_prompt = pattern.sub(lambda m: '*' * 12 + m.group(2), context.rendered_prompt)


# Pydantic Model for Request Body
class ChatRequest(BaseModel):
    message: str

# For Structured Output
class Step(KernelBaseModel):
    message: str

class Utilities:
    """A utility class which will have multiple functions which can be used by the Agents."""

    @kernel_function(description="Checks if the given JSON string is valid and adheres to the structure defined by the class Step")
    def validate_json(self, json_str: str) -> bool:
        """
        Attempts to parse the given JSON string into a Step instance.
        Returns True if the JSON adheres to the Step structure, otherwise False.
        """
        try:
            # Attempt to parse the JSON string into a Step instance.
            step = Step.model_validate_json(json_str)
            # print("Valid JSON. Parsed Step instance:", step)
            print("From the plug-in: Valid JSON")
            return True
        except Exception as e:  # Catching any exception since we don't have access to a specific ValidationError
            print("From the plug-in: Invalid JSON or schema mismatch:", e)
            return False

    @kernel_function(description="Masks a bank account number in a given string by replacing the first six digits with asterisks, assuming the number is 10 digits long")
    def mask_bank_account(self, input_str: str) -> str:
        """
        Searches for a bank account number in the given string and masks it.
        Assumes that a bank account number is exactly 10 digits long.
        The first 6 digits are replaced with '*' while the last 4 digits remain visible.
        """
        import re
        # Regex pattern to match a 10-digit bank account number (grouping first 6 and last 4 digits)
        pattern = re.compile(r'\b(\d{6})(\d{4})\b')
        
        # Replace the account number with masked version: first 6 digits are replaced with '*' 
        masked_str = pattern.sub(lambda m: '*' * 6 + m.group(2), input_str)
        
        print("From the plugin: Masked account number string:", masked_str)
        return masked_str


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

@app.post("/agent5")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent5's response.
    """
    AGENT5_SERVICEID = "agent5"

    kernel.add_service(AzureChatCompletion(service_id=AGENT5_SERVICEID))
    kernel.add_plugin(Utilities(), plugin_name="utilities")
    agent5Settings = kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT5_SERVICEID)
    agent5Settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    agent5Settings.response_format=Step

    # Define the Agent
    AGENT_NAME = "Agent5"
    AGENT_INSTRUCTIONS = """You are a helpful intelligent agent that repeats the user message as an agent no.5 from the movie The Matrix. Please mention the agent number in your response.
    You will invoke the LLM to fetch the response and then make use of the function validate_json to validate the JSON structure of the response. 
    If validate_json returns True, it means that the JSON is valid. If it returns False, the JSON is invalid.
    If the JSON structure is invalid, you will prompt the LLM to correct the response. If the JSON structure doesnt get corrected in 3 attempts respond with an error message - All agents are busy, please try later."""
    #You will attempt to correct the response 3 times based on the output of validate_json. If the correction attempts exceed 3 and your are still not able to fetch the response in the correct JSON schema, you will send an error message."""
    agent = ChatCompletionAgent(service_id=AGENT5_SERVICEID, kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS, arguments=KernelArguments(settings=agent5Settings))

    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    response_text = ""
    async for content in agent.invoke(chat_history):
        # step = Step.model_validate_json(content.content)
        chat_history.add_message(content) # Store the last response
        response_text = content.content  

    return {"user_input": request.message, "agent_response": response_text}


@app.post("/agent6")
async def chat_with_agent(request: ChatRequest):
    """
    Handles user input and returns the agent6's response.
    This version uses a prompt function so that the PROMPT_RENDERING filters (e.g. masking) are applied,
    and it passes the agent instructions to the LLM.
    """
    AGENT6_SERVICEID = "agent6"

    # Define your agent instructions.
    AGENT_INSTRUCTIONS = (
        "You are a helpful intelligent agent that repeats the user message as an agent no.6 from the movie The Matrix. "
        "Please mention the agent number in your response. "
        "You will invoke the LLM to fetch the response and then make use of the function mask_bank_account to mask the bank account number in the response."
    )

    # Add the Azure Chat Completion service and your utilities plugin.
    kernel.add_service(AzureChatCompletion(service_id=AGENT6_SERVICEID))
    kernel.add_plugin(Utilities(), plugin_name="utilities")
    agent6Settings = kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT6_SERVICEID)
    agent6Settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    # agent6Settings.response_format = Step

    # Create a prompt function that includes placeholders for agent_instructions, chat_history, and user_input.
    # Note the use of '$' before the variable names to indicate these are variables.
    chat_function_agent6 = kernel.add_function(
        plugin_name="Agent6Chat",
        function_name="Chat",
        prompt="{{$agent_instructions}}\n{{$chat_history}}{{$user_input}}",
        template_format="semantic-kernel",
        prompt_execution_settings=agent6Settings,
    )

    # Prepare the chat history.
    chat_history = ChatHistory()
    chat_history.add_user_message(request.message)

    # When invoking, pass the agent_instructions along with chat_history and user_input.
    result = await kernel.invoke(
        chat_function_agent6,
        KernelArguments(
            chat_history=chat_history,
            user_input=request.message,
            agent_instructions=AGENT_INSTRUCTIONS,
        )
    )

    # The rendered prompt (with filtering applied) is used by the LLM and the final response is returned.
    return {"user_input": request.message, "agent_response": str(result)}





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
