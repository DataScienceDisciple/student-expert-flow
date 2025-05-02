import json  # Import the json library
# Correct import from the SDK & Add WebSearchTool
from agents import Agent, WebSearchTool
from student_expert_flow.config import ExpertConfig, StudentConfig
# Import the structured output model
from student_expert_flow.models import StudentOutput


class ExpertAgent:
    def __init__(self, config: ExpertConfig):
        """Initializes the Expert Agent using configuration."""
        self.config = config
        # Default instructions to supplement base instructions
        default_instructions = "You are an expert providing authoritative explanations. Be clear and concise."

        # Combine instructions - ensuring newline characters are correctly handled
        effective_instructions = f"{default_instructions}\n\n{config.instructions}"

        # Initialize the underlying agent from the SDK
        # Passing parameters based on the config object
        self.agent = Agent(
            name=config.name,
            instructions=effective_instructions,
            model=config.model,
            tools=[WebSearchTool()]  # Add WebSearchTool here
            # max_tokens might be implicitly handled by the SDK or set elsewhere?
            # For now, we'll omit it unless explicitly required by Agent signature
            # Or perhaps it's part of a ModelSettings object?
            # Revisit if initialization fails or behaves unexpectedly.
            # max_tokens=config.max_tokens
            # Tools will be added later
        )
        print(f"Expert Agent '{self.config.name}' initialized.")

    # Removed placeholder respond method - Runner will invoke self.agent


class StudentAgent:
    def __init__(self, config: StudentConfig):
        """Initializes the Student Agent using configuration and structured output."""
        self.config = config

        # Generate the schema dictionary
        schema_dict = StudentOutput.model_json_schema()
        # Convert schema dictionary to indented JSON string
        schema_json_string = json.dumps(schema_dict, indent=2)

        # Update instructions for structured output using the indented string
        structured_output_instructions = (
            "Your response MUST be a JSON object matching the following Pydantic schema:\n"
            f"```json\n{schema_json_string}\n```\n"
            "Base your response_content and is_goal_achieved assessment on the ongoing conversation context "
            "and your primary learning goal.\n"
            f"Your ultimate learning goal is: {config.goal}"
        )

        default_instructions = (
            "You are a student trying to achieve a specific learning goal.\n"
            "Ask clear questions to the expert to gather the information you need.\n"
            "Evaluate the expert's responses against your goal.\n"
            "Use the required JSON format for your response."
        )
        effective_instructions = f"{structured_output_instructions}\n\n{default_instructions}\n\n{config.instructions}"

        # Initialize the underlying agent with the specified output_type
        self.agent = Agent(
            name=config.name,
            instructions=effective_instructions,
            output_type=StudentOutput,  # Specify the Pydantic model here
            model=config.model  # Pass the configured model
        )
        print(
            f"Student Agent '{self.config.name}' initialized with goal: '{self.config.goal}' using model '{config.model}' (structured output)."
        )

    # Removed is_goal_achieved method - logic moved to runner checking the structured output

    # Removed placeholder ask method - Runner will invoke self.agent
