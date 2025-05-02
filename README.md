# Student-Expert Flow

A project demonstrating a dialogue flow between a 'student' agent learning a topic and an 'expert' agent providing information, using the openai-agents-python library.

## Setup

1. Clone the repository.
2. Ensure you have Poetry installed. If not, follow the installation instructions [here](https://python-poetry.org/docs/#installation).
3. Navigate to the project directory and install dependencies:
   ```bash
   poetry install
   ```
4. Activate the virtual environment managed by Poetry:
   ```bash
   poetry shell
   ```

## Configuration

Agent behavior is configured using YAML files located in the `configs/` directory.

- `expert_config.yaml`: Defines the parameters for the Expert agent (e.g., instructions, model).
- `student_config.yaml`: Defines the parameters for the Student agent (e.g., instructions, goal, max iterations).

Sample configurations are provided. You can modify these or create new ones to customize the dialogue.

## Usage

(Instructions to be added)
