import asyncio  # Import asyncio if we anticipate using Runner.run
from typing import List, Dict, Any

from student_expert_flow.agents import StudentAgent, ExpertAgent
from agents import Runner, Agent  # Import Runner and base Agent
# Assuming RunResult might be needed for type hinting or checking, but not strictly necessary
# from agents.result import RunResult
# Import the structured output model
from student_expert_flow.models import StudentOutput


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
            expert_result = await Runner.run(expert.agent, input=conversation_input)
            # Ensure it's a string
            expert_response = str(expert_result.final_output)
            print(f"Expert ({expert.config.name}): {expert_response}")

            # Add expert response to full history log
            full_history.append(
                {"role": "assistant", "agent": expert.config.name, "content": expert_response})

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
            f"[{entry.get('agent', 'System')} ({entry['role']})]: {entry['content']}")

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
