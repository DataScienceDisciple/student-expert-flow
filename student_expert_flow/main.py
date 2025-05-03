import argparse
import asyncio
import logging
import os
from dotenv import load_dotenv

# Import necessary components from the project
from student_expert_flow.participants import StudentAgent, ExpertAgent
from student_expert_flow.runner import run_dialogue
from student_expert_flow.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def async_main():
    parser = argparse.ArgumentParser(
        description="Run a dialogue between a Student and an Expert agent.")
    parser.add_argument("--student-config", required=True,
                        help="Path to the Student agent's YAML configuration file.")
    parser.add_argument("--expert-config", required=True,
                        help="Path to the Expert agent's YAML configuration file.")
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum number of dialogue turns.")
    parser.add_argument("--output-dir", default="transcripts",
                        help="Directory to save conversation transcripts and summaries.")
    # Add a verbose flag later if needed (Task 11)

    args = parser.parse_args()

    try:
        # 1. Load Configs
        logger.info(f"Loading student config from: {args.student_config}")
        student_config_data = load_config(args.student_config, 'student')
        if not student_config_data:
            logger.error(
                f"Failed to load student config from {args.student_config}")
            return  # Exit if config loading failed

        logger.info(f"Loading expert config from: {args.expert_config}")
        expert_config_data = load_config(args.expert_config, 'expert')
        if not expert_config_data:
            logger.error(
                f"Failed to load expert config from {args.expert_config}")
            return  # Exit if config loading failed

        # Ensure output directory exists (used by transcript saving in runner)
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {args.output_dir}")
        # Note: The runner currently saves transcripts to a hardcoded dir relative to its own location.
        # We need to modify the runner to accept the output directory or handle path construction here.
        # For now, just creating the directory based on the argument.

        # 2. Initialize Agents
        logger.info("Initializing agents...")
        # Pass the loaded config *objects* to the agent constructors
        student = StudentAgent(student_config_data)
        expert = ExpertAgent(expert_config_data)
        logger.info(
            f"Student Agent: {student.config.name}, Goal: {student.config.goal}")
        logger.info(
            f"Expert Agent: {expert.config.name}, Model: {expert.config.model}")

        # 3. Run Dialogue
        logger.info(f"Starting dialogue with max turns: {args.max_turns}")
        await run_dialogue(student, expert, max_turns=args.max_turns)
        # The run_dialogue function now handles transcript/summary saving.
        # We might need to pass args.output_dir into it later if we centralize output path handling.

        logger.info("Dialogue finished.")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
    except Exception as e:
        # Log traceback
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        # Consider returning an error code or raising exception for the caller


def main():
    # Load environment variables from .env file *before* anything else
    # Set override=True to ensure .env values take precedence over existing env vars
    load_dotenv(override=True)

    # Setup asyncio event loop and run main coroutine
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
        # Optionally return a non-zero exit code
    except Exception as e:
        # Catch any other unexpected errors during startup/asyncio run
        logger.critical(
            f"Critical error during script execution: {e}", exc_info=True)
        # Optionally return a non-zero exit code


if __name__ == "__main__":
    main()
