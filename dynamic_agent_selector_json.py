"""
This Python script demonstrates the use of **Semantic Kernel Agents** in an **Agent Group Chat**
to simulate a conversational AI system where multiple agents collaborate.

Key functionalities:
- The **Orchestrator Agent** determines the user's mood (serious or funny) based on input.
- A **MoodSetterPlugin** updates a global variable (`is_mood_serious`) to track the detected mood.
- Based on the user's mood:
  - If serious, **Agent1** (serious agent) is chosen.
  - If not serious, **Agent2** (funny agent) is chosen.
- The **Agent Leader** initiates a conversation with the selected agent about how to protect the Matrix.
- The conversation continues until the **ApprovalTerminationStrategy** detects that the response is "approved."

The script uses **Azure OpenAI** for chat completion and manages the conversation flow asynchronously using **async/await**.
"""

import asyncio
import json # For JSON parsing
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

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel # Introduced for Structured Output


# Telemetry setup code goes here

# Replace the connection string with your Application Insights connection string
connection_string = "<App-Insights-Connection-String-Here>"

# Create a resource to represent the service/sample
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "dynamic-agent-selector"})


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

# For Structured Output
class Step(KernelBaseModel):
    protectionSteps: str
    


def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(service_id=service_id))
    return kernel


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "approved" in history[-1].content.lower()

is_mood_serious = False

# Define a sample plugin for the sample
class MoodSetterPlugin:
    """A plugin to set the is_mood_serious variable."""

    @kernel_function(description="Sets the mood of the user by setting a boolean flag is_mood_serious depicting if the mood is serious or not.")
    def set_mood(self, isMoodSerious):
        global is_mood_serious
        print(f"Setting mood to serious: {isMoodSerious}")
        is_mood_serious = isMoodSerious

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
            print("Valid JSON. Parsed Step instance:", step)
            return True
        except Exception as e:  # Catching any exception since we don't have access to a specific ValidationError
            print("Invalid JSON or schema mismatch:", e)
            return False


async def main():

    ORCHESTRATOR_NAME = "Orchestrator"
    ORCHESTRATOR_INSTRUCTIONS = """
    You are the Orchestrator. You are responsible for setting the mood of the user. You can set the mood to serious or not serious depending on the mood of the user input statement.
    """
    orchestrator_service_id = "orchestrator"
    orchestrator_kernel = _create_kernel_with_chat_completion(orchestrator_service_id)
    orchestrator_kernel.add_plugin(MoodSetterPlugin(),plugin_name="moodsetter")
    settings = orchestrator_kernel.get_prompt_execution_settings_from_service_id(service_id=orchestrator_service_id)
    # Configure the function choice behavior to auto invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    orchestrator = ChatCompletionAgent(
        service_id=orchestrator_service_id,
        kernel=orchestrator_kernel,
        name=ORCHESTRATOR_NAME,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        arguments=KernelArguments(settings=settings)
    )

    # User input
    user_input = "I feel so serious."
    chat_history = ChatHistory()
    chat_history.add_user_message(user_input)

    agent_name: str | None = None

    async for content in orchestrator.invoke_stream(chat_history):
        if not agent_name:
            agent_name = content.name
            print(f"{agent_name}: '", end="")
        if (
            not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)
            and content.content.strip()
        ):
            print(f"{content.content}", end="", flush=True)
    
    print("****")
    print(f"Is mood serious: {is_mood_serious}")

    AGENT_LEADER_NAME = "AgentLeader"
    AGENT_LEADER_INSTRUCTIONS = """
    You are the Agent Leader guarding the Matrix. You have 2 Agents working for you. The main job of you and the agents is to protect the Matrix.
    You are responsible for initiating a conversation with one of the 2 Agents - Agent1 and Agent2.
    Agent1 is the serious agent and Agent2 is a funny agent.
    You will have a conversation with the chosen Agent as to how that agent plans to protect the Matrix. You should continue the conversation till you are satisfied with the response from the Agent.
    You should also validate if the response from the Agent is a valid JSON string and adheres to the structure defined by the class Step - example: 
    {"protectionSteps": "Ensure proper network segmentation, enforce strong authentication, and maintain up-to-date firewall configurations."}
    If you are satisfied with the response and the response from the Agent adheres to the structure defined by the class Step, state that it is approved.
    If not, provide insight on how the Agent can improve on their suggestion. DO NOT approve if the Agent's response doesnt adhere to the structure defined by the class Step.
    """
    agent_leader_service_id = "agentleader"
    agent_leader_kernel = _create_kernel_with_chat_completion(agent_leader_service_id)
    agent_leader_kernel.add_plugin(Utilities(), plugin_name="utilities")
    agentLeaderSettings = agent_leader_kernel.get_prompt_execution_settings_from_service_id(service_id=agent_leader_service_id)
    # Configure the function choice behavior to auto invoke kernel functions
    agentLeaderSettings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    agent_leader = ChatCompletionAgent(
        service_id=agent_leader_service_id,
        kernel=agent_leader_kernel,
        name=AGENT_LEADER_NAME,
        instructions=AGENT_LEADER_INSTRUCTIONS,
        arguments=KernelArguments(settings=agentLeaderSettings)
    )

    AGENT1_NAME = "Agent1"
    AGENT1_SERVICEID = "agent1"
    AGENT1_INSTRUCTIONS = """
    You are a very serious agent protecting the Matrix. You are Agent1. You are responsible for protecting the Matrix from any threats. You should provide a serious response to the Agent Leader.
    You should provide the Agent Leader with a plan on how you plan to protect the Matrix. You should continue the conversation till the Agent Leader is satisfied with your response.
    You should mention the agent number in your response. 
    """
    agent1_kernel = _create_kernel_with_chat_completion(AGENT1_SERVICEID)
    agent1Settings = agent1_kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT1_SERVICEID)
    agent1Settings.response_format=Step

    agent1 = ChatCompletionAgent(
        service_id=AGENT1_SERVICEID,
        kernel=agent1_kernel,
        name=AGENT1_NAME,
        instructions=AGENT1_INSTRUCTIONS,
        arguments=KernelArguments(settings=agent1Settings)
    )
    

    AGENT2_NAME = "Agent2"
    AGENT2_SERVICEID = "agent2"
    AGENT2_INSTRUCTIONS = """
    You are a very funny agent protecting the Matrix. You are Agent2. You are responsible for protecting the Matrix from any threats. You should provide a funny response to the Agent Leader.
    You should provide the Agent Leader with a plan on how you plan to protect the Matrix. You should continue the conversation till the Agent Leader is satisfied with your response.
    You should mention the agent number in your response. 
    """
    agent2_kernel = _create_kernel_with_chat_completion(AGENT2_SERVICEID)
    agent2Settings = agent2_kernel.get_prompt_execution_settings_from_service_id(service_id=AGENT2_SERVICEID)
    agent2Settings.response_format=Step

    agent2 = ChatCompletionAgent(
        service_id=AGENT2_SERVICEID,
        kernel=agent2_kernel,
        name=AGENT2_NAME,
        instructions=AGENT2_INSTRUCTIONS,
        arguments=KernelArguments(settings=agent2Settings)
    )


    if is_mood_serious:
        group_chat = AgentGroupChat(
            agents=[
                agent1,
                agent_leader
            ],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[agent_leader],
                maximum_iterations=10,
            ),
        )
    elif not is_mood_serious:
        group_chat = AgentGroupChat(
            agents=[
                agent2,
                agent_leader
            ],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[agent_leader],
                maximum_iterations=10,
            ),
        )

    input = "how to best protect the Matrix."

    await group_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=input))
    print(f"# User: '{input}'")

    async for content in group_chat.invoke():
        print(f"# Agent - {content.name or '*'}: '{content.content}'")

    print(f"# IS COMPLETE: {group_chat.is_complete}")


if __name__ == "__main__":
    asyncio.run(main())
