from pydantic import BaseModel, Field


class StudentOutput(BaseModel):
    """Structured output for the Student agent."""
    is_goal_achieved: bool = Field(
        ..., description="Set to true if the learning goal has been fully met based on the conversation, false otherwise.")
    response_content: str = Field(
        ..., description="The student's response, question, or statement based on the current dialogue state.")

    # Add model_config example if needed for specific JSON schema generation
    # class Config:
    #     json_schema_extra = {
    #         "examples": [
    #             {
    #                 "is_goal_achieved": False,
    #                 "response_content": "Thanks, that helps! Can you explain the difference between X and Y?"
    #             },
    #             {
    #                 "is_goal_achieved": True,
    #                 "response_content": "Excellent, I understand the concept now."
    #             }
    #         ]
    #     }
