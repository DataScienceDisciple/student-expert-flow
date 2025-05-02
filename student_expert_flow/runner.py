import asyncio  # Import asyncio if we anticipate using Runner.run
from typing import List, Dict, Any
import logging

from student_expert_flow.agents import StudentAgent, ExpertAgent
from agents import Runner, Agent  # Import Runner and base Agent
# Import the specific result type for type hinting
from agents.result import RunResult
# Import item and response types needed for checking citations/tool calls
from agents.items import MessageOutputItem, ToolCallItem
from openai.types.responses.response_output_text import (
    ResponseOutputText,
    AnnotationURLCitation,
)
# Import the structured output model
from student_expert_flow.models import StudentOutput

# Add logger
logger = logging.getLogger(__name__)


async def run_dialogue(student: StudentAgent, expert: ExpertAgent, max_turns: int = 5):
    """Runs a dialogue loop between a Student and an Expert agent using agents.Runner."""

    print(f"--- Starting Dialogue --- Goal: {student.config.goal} ---")

    # Initialize conversation history. Start with the student's goal perhaps?
    # The format expected by Runner.run is List[Dict[str, Any]] usually like
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # We'll start the student turn with the initial goal.
    conversation_input: List[Dict[str, Any]] = [
        {"role": "user", "content": f"My learning goal is: {student.config.goal}. Please explain the basics and respond in the required JSON format."}
    ]
    full_history: List[Dict[str, Any]] = list(
        conversation_input)  # Keep a separate log

    current_turn = 0

    while current_turn < max_turns:
        current_turn += 1
        print(f"\n--- Turn {current_turn} ---")

        # 1. Student Turn (Outputs StudentOutput JSON)
        print(
            f"Running Student ({student.config.name})... Input: {conversation_input[-1]['content']}")
        try:
            # Use student.agent which is the actual agents.Agent instance
            student_result = await Runner.run(student.agent, input=conversation_input)

            # --- Process Structured Output ---
            if not isinstance(student_result.final_output, StudentOutput):
                print(
                    f"ERROR: Student agent did not return expected StudentOutput object. Got: {type(student_result.final_output)}")
                # Attempt to use raw string content as fallback?
                student_response_content = str(student_result.final_output)
                goal_achieved = False  # Assume goal not achieved if format is wrong
                print(
                    f"Student ({student.config.name}) [Fallback]: {student_response_content}")
            else:
                student_output: StudentOutput = student_result.final_output
                student_response_content = student_output.response_content
                goal_achieved = student_output.is_goal_achieved
                print(
                    f"Student ({student.config.name}) [Goal Achieved: {goal_achieved}]: {student_response_content}")

            # Add student response to full history log (using the text content)
            full_history.append(
                {"role": "user", "agent": student.config.name, "content": student_response_content, "goal_achieved_flag": goal_achieved})

            # Prepare input for the expert turn using the SDK's helper
            # The history should contain the raw LLM response, which might be JSON string
            # Let's ensure the expert sees the student's *response_content* clearly
            conversation_input = student_result.to_input_list()
            # Replace the last assistant message (the JSON output) with just the text content for the expert?
            # Or maybe the expert's prompt should handle the JSON input? Let's keep history raw for now.
            # If expert struggles, we might need to adjust history before passing to expert.
            # Let's add the student's text response as a user message for clarity
            conversation_input.append(
                {"role": "user", "content": student_response_content})

        except Exception as e:
            print(f"Error during Student turn: {e}")
            break  # Exit loop on error

        # Check goal achievement based on the structured output flag
        if goal_achieved:
            print(f"--- Dialogue End (Goal Achieved according to Student) ---")
            break

        # Ensure conversation_input is not empty before expert turn
        if not conversation_input:
            print("Error: Conversation input became empty after student turn.")
            break

        # 2. Expert Turn (Still outputs text)
        print(
            f"Running Expert ({expert.config.name})... Input: {conversation_input[-1]['content']}")
        try:
            # Use expert.agent which is the actual agents.Agent instance
            expert_result: RunResult = await Runner.run(expert.agent, input=conversation_input)
            # Ensure it's a string
            expert_response = str(expert_result.final_output)
            print(f"Expert ({expert.config.name}): {expert_response}")

            # --- Check if web search results (citations) were used ---
            expert_used_web_search_this_turn = False
            logger.debug(
                f"--- Checking expert result items for web_search_call ---")
            for idx, item in enumerate(expert_result.new_items):
                logger.debug(
                    f"Item {idx}: Type={type(item)}, RawItemType={getattr(item.raw_item, 'type', 'N/A')}")
                # Check if it's a ToolCallItem indicating a web search was performed
                if isinstance(item, ToolCallItem) and hasattr(item.raw_item, 'type') and item.raw_item.type == 'web_search_call':
                    logger.debug(
                        f"  Item {idx} is ToolCallItem with type 'web_search_call'. Setting flag to True.")
                    expert_used_web_search_this_turn = True
                    break  # Found the tool call, no need to check further

                # Optional: Keep the citation check as a secondary confirmation or for future use?
                # For now, prioritizing the tool call itself.
                # if isinstance(item, MessageOutputItem) and hasattr(item.raw_item, 'role') and item.raw_item.role == 'assistant':
                #     logger.debug(f"  Item {idx} is Assistant MessageOutputItem.")
                #     if isinstance(item.raw_item.content, list):
                #         logger.debug(f"  Item {idx} content is a list.")
                #         for content_idx, content_part in enumerate(item.raw_item.content):
                #             logger.debug(f"    Content part {content_idx}: Type={type(content_part)}")
                #             if isinstance(content_part, ResponseOutputText) and hasattr(content_part, 'annotations') and isinstance(content_part.annotations, list):
                #                 logger.debug(f"      Content part {content_idx} is ResponseOutputText with annotations list.")
                #                 if not content_part.annotations:
                #                     logger.debug(f"        Annotations list is empty.")
                #                 else:
                #                     for ann_idx, annotation in enumerate(content_part.annotations):
                #                         logger.debug(f"        Annotation {ann_idx}: Type={type(annotation)}")
                #                         if isinstance(annotation, AnnotationURLCitation):
                #                             logger.debug(f"          Found AnnotationURLCitation! Setting flag to True.")
                #                             # expert_used_web_search_this_turn = True # Flag set by ToolCallItem now
                #                             break # Found one citation, enough to confirm
                #             elif isinstance(content_part, ResponseOutputText):
                #                 logger.debug(f"      Content part {content_idx} is ResponseOutputText but has no 'annotations' list attribute or it's not a list.")
                #             else:
                #                 logger.debug(f"      Content part {content_idx} is not ResponseOutputText.")
                #             # if expert_used_web_search_this_turn: break # Break is handled by ToolCallItem check
                #     else:
                #         logger.debug(f"  Item {idx} content is not a list (Type: {type(item.raw_item.content)}). Skipping annotation check for this item.")
                # else:
                #      logger.debug(f"  Item {idx} is not an Assistant MessageOutputItem or ToolCallItem.")
                # # if expert_used_web_search_this_turn: break # Break is handled by ToolCallItem check

            logger.debug(
                f"--- Finished checking items. Used web search this turn: {expert_used_web_search_this_turn} ---")
            # --- End check ---

            # Add expert response to full history log, including web search usage flag
            full_history.append(
                {"role": "assistant", "agent": expert.config.name, "content": expert_response, "used_web_search": expert_used_web_search_this_turn})

            # Prepare input for the next student turn
            conversation_input = expert_result.to_input_list()

        except Exception as e:
            print(f"Error during Expert turn: {e}")
            break  # Exit loop on error

        # Ensure conversation_input is not empty before next student turn
        if not conversation_input:
            print("Error: Conversation input became empty after expert turn.")
            break

    else:  # Loop finished without break (max_turns reached)
        print(f"\n--- Dialogue End (Max Turns Reached: {max_turns}) ---")

    print("\n--- Full Conversation History Log ---")
    for entry in full_history:
        # Adjusting print format slightly for clarity
        print(
            f"[{entry.get('agent', 'System')} ({entry['role']})]: {entry['content']}" +
            (f" [Used Web Search: {entry.get('used_web_search')}]" if 'used_web_search' in entry else "")
        )

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
