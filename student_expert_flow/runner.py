import asyncio  # Import asyncio if we anticipate using Runner.run
from typing import List, Dict, Any
import logging
import os  # Import os for path manipulation

from student_expert_flow.participants import StudentAgent, ExpertAgent
from agents import Runner, Agent  # Import Runner and base Agent
# Import the specific result type for type hinting
from agents.result import RunResult
# Import item and response types needed for checking citations/tool calls
from agents.items import ToolCallItem

# Import the structured output model
from student_expert_flow.models import StudentOutput
# Import transcript saving function
from .transcript import save_transcript, format_transcript, generate_summary

# Add logger
logger = logging.getLogger(__name__)


async def run_dialogue(student: StudentAgent, expert: ExpertAgent, max_turns: int = 5, output_dir: str = "transcripts"):
    """Runs a dialogue loop between a Student and an Expert agent using agents.Runner.

    The flow is: System Goal -> Expert -> Student -> Expert -> Student ...

    Args:
        student: The initialized StudentAgent.
        expert: The initialized ExpertAgent.
        max_turns: Maximum number of turns for the dialogue.
        output_dir: Directory to save transcript and summary files.
    """

    logger.info(
        f"--- Starting Dialogue --- Goal: {student.config.goal} --- Max Turns: {max_turns} ---")

    # Initialize conversation input with the student's goal, framed as a user request to the expert.
    initial_message = {
        "role": "user", "content": f"My learning goal is: {student.config.goal}. Please provide an initial explanation or ask clarifying questions."}
    # This list is passed to the Runner, must only contain valid keys (role, content, id)
    conversation_input: List[Dict[str, Any]] = [initial_message]

    # Keep a separate log, starting with the same initial message but add the agent key for transcript clarity.
    # Make a copy to avoid modifying the dict shared with conversation_input initially
    logged_initial_message = initial_message.copy()
    logged_initial_message['agent'] = 'System'
    full_history: List[Dict[str, Any]] = [logged_initial_message]

    current_turn = 0
    goal_achieved = False  # Initialize goal achievement status

    while current_turn < max_turns:
        current_turn += 1
        logger.info(f"--- Turn {current_turn} ---")

        # --- Expert Turn --- #
        # Ensure conversation_input is not empty before expert turn
        if not conversation_input:
            logger.error(
                "Error: Conversation input became empty before expert turn.")
            break

        logger.info(
            f"Running Expert ({expert.config.name})... Input: {conversation_input[-1]['content']}")
        try:
            expert_result: RunResult = await Runner.run(expert.agent, input=conversation_input)
            # Ensure it's a string
            expert_response = str(expert_result.final_output)
            logger.info(f"Expert ({expert.config.name}): {expert_response}")

            # Check for web search tool usage
            expert_used_web_search_this_turn = False
            logger.debug(
                f"--- Checking expert result items for web_search_call ---")
            for idx, item in enumerate(expert_result.new_items):
                logger.debug(
                    f"Item {idx}: Type={type(item)}, RawItemType={getattr(item.raw_item, 'type', 'N/A')}")
                if isinstance(item, ToolCallItem) and hasattr(item.raw_item, 'type') and item.raw_item.type == 'web_search_call':
                    logger.debug(
                        f"  Item {idx} is ToolCallItem with type 'web_search_call'. Setting flag to True.")
                    expert_used_web_search_this_turn = True
                    break
            logger.debug(
                f"--- Finished checking items. Used web search this turn: {expert_used_web_search_this_turn} ---")

            # Add expert response to full history log
            full_history.append(
                {"role": "assistant", "agent": expert.config.name, "content": expert_response, "used_web_search": expert_used_web_search_this_turn})

            # Prepare input for the student turn
            # Get history including the expert's assistant-role response
            conversation_input = expert_result.to_input_list()
            # **Architect Fix 2 (Revised):** Append the Expert's text response as a new user message
            # This avoids mutating the role/content type of the SDK-generated item.
            if expert_response:  # Avoid adding empty messages
                conversation_input.append(
                    {"role": "user", "content": expert_response})

        except Exception as e:
            logger.error(f"Error during Expert turn {current_turn}: {e}")
            # Optionally add a user-facing print here? For now, rely on logger.
            # print(f"Error during Expert turn {current_turn}: {e}")
            break  # Exit loop on error

        # --- Check for Max Turns AFTER Expert --- #
        if current_turn == max_turns:
            logger.info(
                f"--- Dialogue End (Max Turns Reached: {max_turns}) --- ")
            break  # Exit loop before the final student turn

        # --- Student Turn --- #
        # Ensure conversation_input is not empty before student turn
        if not conversation_input:
            logger.error(
                "Error: Conversation input became empty before student turn.")
            break

        logger.info(
            f"Running Student ({student.config.name})... Input: {conversation_input[-1]['content']}")
        try:
            student_result: RunResult = await Runner.run(student.agent, input=conversation_input)

            # Process structured output (or fallback)
            if not isinstance(student_result.final_output, StudentOutput):
                logger.warning(
                    f"Student agent did not return expected StudentOutput object. Got: {type(student_result.final_output)}")
                student_response_content = str(student_result.final_output)
                goal_achieved = False  # Assume goal not achieved if format is wrong
                logger.info(
                    f"Student ({student.config.name}) [Fallback]: {student_response_content}")
            else:
                student_output: StudentOutput = student_result.final_output
                student_response_content = student_output.response_content
                goal_achieved = student_output.is_goal_achieved
                logger.info(
                    f"Student ({student.config.name}) [Goal Achieved: {goal_achieved}]: {student_response_content}")

            # Add student response to full history log
            full_history.append(
                {"role": "user", "agent": student.config.name, "content": student_response_content, "goal_achieved_flag": goal_achieved})

            # Prepare input for the next expert turn
            # Get the history list including the student's assistant-role structured output
            conversation_input = student_result.to_input_list()
            # **Architect Fix:** Explicitly add the student's text response as a user message
            # This ensures the Expert agent sees the student's last message as user input.
            if student_response_content:  # Avoid adding empty messages
                conversation_input.append(
                    {"role": "user", "content": student_response_content})

        except Exception as e:
            logger.error(f"Error during Student turn {current_turn}: {e}")
            # Optionally add a user-facing print here? For now, rely on logger.
            # print(f"Error during Student turn {current_turn}: {e}")
            break  # Exit loop on error

        # Check for goal achievement AFTER student turn
        if goal_achieved:
            logger.info(
                f"--- Dialogue End (Goal Achieved according to Student on Turn {current_turn}) --- ")
            break

    else:  # Loop finished without break (max_turns reached)
        logger.info(f"--- Dialogue End (Max Turns Reached: {max_turns}) --- ")

    # Log the full history at DEBUG level instead of printing
    logger.debug("--- Full Conversation History Log ---")
    for entry in full_history:
        logger.debug(
            f"[{entry.get('agent', 'System')} ({entry['role']})]: {entry['content']}" +
            (f" [Used Web Search: {entry.get('used_web_search')}]" if 'used_web_search' in entry else "") +
            # Also log goal flag
            (f" [Goal Achieved Flag: {entry.get('goal_achieved_flag')}]" if 'goal_achieved_flag' in entry else "")
        )

    # --- Save Transcript --- #
    transcript_path = None  # Initialize path
    formatted_transcript = ""
    try:
        # Format first, as it's needed for both saving and summarizing
        formatted_transcript = format_transcript(
            full_history, student.config.goal)
        transcript_path = save_transcript(
            history=full_history,
            goal=student.config.goal,
            formatted_transcript=formatted_transcript,
            output_dir=output_dir
        )
        logger.info(f"Transcript saved to Markdown: {transcript_path}")

        # --- Generate and Save Summary --- #
        if transcript_path and formatted_transcript:
            try:
                # Use expert's model for summary
                summary = await generate_summary(formatted_transcript, model=expert.config.model)
                summary_filename = os.path.splitext(transcript_path)[
                    0] + ".summary.txt"
                with open(summary_filename, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Summary saved to {summary_filename}")
            except Exception as summary_e:
                logger.error(
                    f"Failed to generate or save summary: {summary_e}")
        else:
            logger.warning(
                "Skipping summary generation because transcript saving failed or content was empty.")
        # --- End Generate and Save Summary --- #

    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
    # --- End Save Transcript ---

    # Return the detailed history we logged
    return full_history

# Example of how this might be called later (e.g., from a main script/CLI)
# async def main():
#     # 1. Load Configs
#     expert_config = load_config("configs/expert_config.yaml", 'expert')
#     student_config = load_config("configs/student_config.yaml", 'student')
#
#     # 2. Initialize Agents
#     expert = ExpertAgent(expert_config)
#     student = StudentAgent(student_config)
#
#     # 3. Run Dialogue
#     history = await run_dialogue(student, expert, max_turns=student_config.max_iterations)
#
#     # 4. Process history (e.g., save to transcript - Task 8)
#
# if __name__ == "__main__":
#     asyncio.run(main())
