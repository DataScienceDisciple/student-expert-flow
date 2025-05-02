# Student-Expert Flow

A project demonstrating a dialogue flow between a 'student' agent learning a topic and an 'expert' agent providing information, using the `openai-agents-python` library.

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

## Project Structure

- `configs/`: Contains YAML configuration files for agents.
  - `expert_config.yaml`: Configuration for the Expert agent.
  - `student_config.yaml`: Configuration for the Student agent.
- `student_expert_flow/`: Main package directory.
  - `config.py`: Loads and validates configuration using Pydantic.
  - `agents.py`: Contains the `ExpertAgent` and `StudentAgent` class implementations.
  - `__init__.py`: Makes the directory a Python package.
- `tests/`: Contains unit and integration tests.
  - `test_config.py`: Tests for configuration loading.
  - `test_agents.py`: Tests for agent initialization and placeholder methods.
- `pyproject.toml`: Poetry project configuration and dependencies.
- `README.md`: This file.

## Configuration

Agent behavior is configured using YAML files located in the `configs/` directory.

- `expert_config.yaml`: Defines the parameters for the Expert agent (e.g., instructions, model).
- `student_config.yaml`: Defines the parameters for the Student agent (e.g., instructions, goal, max iterations).

Sample configurations are provided. You can modify these or create new ones to customize the dialogue.

## Usage

Currently, the project includes the core agent implementations and configuration loading. A dialogue runner and CLI interface will be added in subsequent steps.

To run tests:

```bash
poetry run pytest
```
