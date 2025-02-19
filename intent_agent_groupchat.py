import asyncio
from enum import Enum

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


# Telemetry setup code goes here

# Replace the connection string with your Application Insights connection string
connection_string = "<App-Insights-Connection-String-Here>"

# Create a resource to represent the service/sample
resource = Resource.create({ResourceAttributes.SERVICE_NAME: "intent-agent-groupchat"})


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


def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(service_id=service_id))
    return kernel


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "approved" in history[-1].content.lower()


class Product(Enum):
    CASA = 1
    Loan = 2
    CreditCards = 3
    NA = 4

product = Product.NA

# Define a plugin for setting the product type
class ProductSetterPlugin:
    """A plugin to set the product variable."""

    @kernel_function(description="Sets the value of the product variable to the desired value set by the calling program.")
    def set_product(self, productType: Product):
        global product
        print(f"Setting the product to: {productType}")
        product = productType    

async def main():

    ORCHESTRATOR_NAME = "IntentOrchestrator"
    ORCHESTRATOR_INSTRUCTIONS = """
    You are the IntentOrchestrator agent of a Retail Bank. You are responsible for identifying the intent of the user basis their question and then setting the value of the variable product on the basis of the identified intent.
    The product variable is an Enum of type Product and can have one of the following values:
    - CASA (Current Account Savings Account)
    - Loan  (Personal Loan, Home Loan, etc.)
    - CreditCards (Credit Cards)
    - NA (Not Applicable)
    """
    orchestrator_service_id = "intentorchestrator"
    orchestrator_kernel = _create_kernel_with_chat_completion(orchestrator_service_id)
    orchestrator_kernel.add_plugin(ProductSetterPlugin(),plugin_name="productsetter")
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
    # User input for opening a Savings Account
    user_input = """I want to open a Savings Account. My details are as follows: 
    Name: John Doe, Age: 25, Aadhar: 123456789, DL: 987654321, Qualification: 12th pass, Initial Deposit: 10000"""


    # User input for applying for a Personal Loan
    # user_input = """
    # Applicant Details:
    # ------------------
    # Name: Rahul Sharma
    # Age: 30
    # Monthly Income: ₹45,000
    # Employment Status: Full-time employed at ABC Corp
    # Credit Score: 700

    # Identification Documents:
    # - PAN Card: Valid (PAN: ABCD1234E)
    # - Aadhar Card: Valid (Aadhar Number: 1234-5678-9012)

    # Financial Documents:
    # - Bank Statement (last 3 months)
    # - Salary Slips (last 3 months)

    # Loan Details:
    # - Loan Amount Requested: ₹500,000
    # - Loan Purpose: Home renovation"""


    # User input for applying for a Credit Card
    # user_input = """
    # Applicant Details:
    # ------------------
    # Name: Priya Verma
    # Age: 28
    # Annual Income: ₹2,00,000
    # Credit Score: 680

    # Identification Documents:
    # - Passport: Valid (Passport Number: P12345678)
    # - Aadhar Card: Valid (Aadhar Number: 9876-5432-1098)

    # Credit Card Details:
    # - Requested Credit Limit: ₹1,00,000
    # - Card Usage: Personal expenses and travel"""


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
    print(f"Identified Product type is: {product}")

    CASA_CHECKER_NAME = "CASAChecker"
    CASA_CHECKER_INSTRUCTIONS = """
        You are the CASA Checker Agent responsible for reviewing and approving the account opening requests initiated by the Maker Agent.
        Your tasks include:
        1. Reviewing the application summary provided by the Maker Agent to verify the accuracy and authenticity of all customer data.
        2. Ensuring that the applicant fully meets the eligibility criteria:
        - The applicant must be at least 18 years of age.
        - The applicant must possess a valid driver's license and a valid Aadhar card.
        - The applicant must have a minimum qualification of being a 10th Standard pass.
        3. Confirming that the Maker Agent has accurately verified and documented these eligibility requirements.
        4. Approving the account opening request if all criteria are met, or providing a detailed explanation and recommendations if any discrepancies or deficiencies are found.
        5. Maintaining the highest standards of data integrity and security during the review process.
        Output should clearly indicate whether the application is approved or requires further action, along with any necessary recommendations.
        Once you are satisfied with the application, you can approve it by mentioning 'approved' in your response.
        """
    casa_checker = ChatCompletionAgent(
        service_id="casachecker",
        kernel=_create_kernel_with_chat_completion("casachecker"),
        name=CASA_CHECKER_NAME,
        instructions=CASA_CHECKER_INSTRUCTIONS,
    )

    CASA_MAKER_NAME = "CASAMaker"
    CASA_MAKER_INSTRUCTIONS = """
    You are the CASA Maker Agent responsible for initiating the account opening process for Current Account and Savings Account (CASA) applications.
    Your tasks include:
    1. Collecting and verifying customer-provided details such as personal information, identification documents, and initial deposit data.
    2. Ensuring that the applicant meets the following eligibility criteria:
    - Must be at least 18 years of age.
    - Must possess a valid driver's license and a valid Aadhar card.
    - Must have a minimum qualification of being a 10th Standard pass.
    3. Verifying that the submitted documents are authentic and the eligibility criteria are clearly documented.
    4. Preparing a comprehensive application summary that includes all collected data and a detailed checklist indicating the eligibility verification.
    5. Logging any anomalies or issues encountered during the data collection process for further investigation.
    Output should be structured, clear, and ready for the next stage of verification by the Checker Agent.
    """
    casa_maker = ChatCompletionAgent(
        service_id="casamaker",
        kernel=_create_kernel_with_chat_completion("casamaker"),
        name=CASA_MAKER_NAME,
        instructions=CASA_MAKER_INSTRUCTIONS,
    )


    LOAN_CHECKER_NAME = "LoanChecker"
    LOAN_CHECKER_INSTRUCTIONS = """
        You are the Loan Checker Agent responsible for reviewing and approving loan applications initiated by the Maker Agent.
        Your tasks include:
        1. Reviewing the application summary provided by the Maker Agent to verify the accuracy and authenticity of all customer data.
        2. Confirming that the applicant meets the defined eligibility criteria:
        - The applicant must be at least 21 years of age.
        - The applicant must have a stable monthly income of at least ₹30,000.
        - The applicant must maintain a satisfactory credit score (e.g., above 650).
        - All required documents (identification, employment details, financial statements) must be present and valid.
        3. Evaluating the Maker Agent’s documentation and highlighting any discrepancies or missing information.
        4. Approving the application if all criteria are met or providing a detailed explanation and recommendations for further action if any deficiencies are found.
        Output should clearly state the approval decision along with any necessary recommendations.
        """
    loan_checker = ChatCompletionAgent(
        service_id="loanchecker",
        kernel=_create_kernel_with_chat_completion("loanchecker"),
        name=LOAN_CHECKER_NAME,
        instructions=LOAN_CHECKER_INSTRUCTIONS,
    )

    LOAN_MAKER_NAME = "LoanMaker"
    LOAN_MAKER_INSTRUCTIONS = """
        You are the Loan Maker Agent responsible for initiating the loan application process.
        Your tasks include:
        1. Collecting and verifying customer-provided details such as personal information, employment data, income details, and credit history.
        2. Ensuring that the applicant meets the following eligibility criteria:
        - Must be at least 21 years of age.
        - Must have a stable monthly income of at least ₹30,000.
        - Must possess a satisfactory credit score (e.g., above 650).
        - Must provide valid identification, employment details, and supporting financial documents (e.g., bank statements).
        3. Preparing a comprehensive application summary that includes all gathered data and a checklist of the eligibility criteria verification.
        4. Logging any anomalies or issues encountered during the data collection process for further investigation.
        Output should be clear, structured, and ready for the Checker Agent to perform an independent review.
        """
    loan_maker = ChatCompletionAgent(
        service_id="loanmaker",
        kernel=_create_kernel_with_chat_completion("loanmaker"),
        name=LOAN_MAKER_NAME,
        instructions=LOAN_MAKER_INSTRUCTIONS,
    )


    CC_CHECKER_NAME = "CCChecker"
    CC_CHECKER_INSTRUCTIONS = """
        You are the Credit Card Checker Agent responsible for reviewing and approving credit card applications initiated by the Maker Agent.
        Your tasks include:
        1. Reviewing the application summary provided by the Maker Agent to verify the accuracy and authenticity of all customer data.
        2. Confirming that the applicant meets the defined eligibility criteria:
        - The applicant must be at least 18 years of age.
        - The applicant must possess a valid identification document (driver's license, passport, or Aadhar card).
        - The applicant must have a minimum annual income of ₹2,50,000.
        - The applicant must maintain a satisfactory credit score as per our requirements.
        3. Validating that the Maker Agent has properly documented all required details and that the eligibility checklist has been fully completed.
        4. Approving the application if all criteria are met, or providing a detailed explanation and recommendations if any discrepancies or missing information are found.
        Output should clearly state whether the application is approved or requires further action, along with any necessary recommendations.
        """
    cc_checker = ChatCompletionAgent(
        service_id="ccchecker",
        kernel=_create_kernel_with_chat_completion("ccchecker"),
        name=CC_CHECKER_NAME,
        instructions=CC_CHECKER_INSTRUCTIONS,
    )

    CC_MAKER_NAME = "CCMaker"
    CC_MAKER_INSTRUCTIONS = """
        You are the Credit Card Maker Agent responsible for initiating the credit card application process.
        Your tasks include:
        1. Collecting and verifying customer-provided details such as personal information, identification data, income details, and credit history.
        2. Ensuring that the applicant meets the following eligibility criteria:
        - Must be at least 18 years of age.
        - Must possess a valid identification document (e.g., driver's license, passport, or Aadhar card).
        - Must have a minimum annual income of ₹2,50,000.
        - Must maintain a satisfactory credit score (as per the bank's requirements).
        3. Preparing a comprehensive application summary that includes all gathered data and a checklist of eligibility criteria verification.
        4. Logging any anomalies or issues encountered during the data collection process for further investigation.
        Output should be structured, clear, and ready for the Checker Agent to perform an independent review.
        """
    cc_maker = ChatCompletionAgent(
        service_id="ccmaker",
        kernel=_create_kernel_with_chat_completion("ccmaker"),
        name=CC_MAKER_NAME,
        instructions=CC_MAKER_INSTRUCTIONS,
    )


    if product == Product.CASA:
        group_chat = AgentGroupChat(
            agents=[
                casa_maker,
                casa_checker
            ],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[casa_checker],
                maximum_iterations=10,
            ),
        )
    elif product == Product.Loan:
        group_chat = AgentGroupChat(
            agents=[
                loan_maker,
                loan_checker
            ],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[loan_checker],
                maximum_iterations=10,
            ),
        )
    elif product == Product.CreditCards:
        group_chat = AgentGroupChat(
            agents=[
                cc_maker,
                cc_checker
            ],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[cc_checker],
                maximum_iterations=10,
            ),
        )


    # input = "how to best protect the Matrix."

    await group_chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
    print(f"# User: '{user_input}'")

    async for content in group_chat.invoke():
        print(f"# Agent - {content.name or '*'}: '{content.content}'")

    print(f"# IS COMPLETE: {group_chat.is_complete}")


if __name__ == "__main__":
    asyncio.run(main())
