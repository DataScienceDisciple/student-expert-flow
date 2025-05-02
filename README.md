# Student-Expert Flow

A project demonstrating a dialogue flow between a 'student' agent learning a topic and an 'expert' agent providing information, using the `openai-agents-python` library. The Student agent uses structured JSON output.

## Setup

1. Clone the repository.
2. Ensure you have Poetry installed. If not, follow the installation instructions [here](https://python-poetry.org/docs/#installation).
3. Navigate to the project directory and install dependencies:
   ```bash
   poetry install
   ```
4. Create a `.env` file in the project root (see Configuration section below) and add your API key.
5. Activate the virtual environment managed by Poetry:
   ```bash
   poetry shell
   ```

## Project Structure

- `configs/`: Contains YAML configuration files for agents.
  - `expert_config.yaml`: Default configuration for the Expert agent.
  - `student_config.yaml`: Default configuration for the Student agent.
  - `expert_config_simple.yaml`: Simple expert config for testing.
  - `student_config_simple.yaml`: Simple student config for testing.
- `student_expert_flow/`: Main package directory.
  - `config.py`: Loads and validates YAML configuration using Pydantic.
  - `models.py`: Defines Pydantic models for structured data (e.g., `StudentOutput`).
  - `agents.py`: Contains the `ExpertAgent` and `StudentAgent` class implementations.
  - `runner.py`: Contains the `run_dialogue` function implementing the conversation flow using `agents.Runner`.
- `tests/`: Contains unit and integration tests.
  - `test_config.py`: Tests for configuration loading.
  - `test_agents.py`: Tests for agent initialization logic.
  - `test_runner.py`: Mocked tests for the dialogue runner logic.
  - `test_runner_integration.py`: Integration tests making real API calls.
- `.env`: (You create this) Environment variables, primarily for API keys.
- `.gitignore`: Specifies intentionally untracked files.
- `pyproject.toml`: Poetry project configuration and dependencies.
- `README.md`: This file.

## Configuration

Agent behavior is configured using YAML files located in the `configs/` directory.

- **API Key:** The primary way to provide the API key (e.g., `OPENAI_API_KEY`) is via a `.env` file in the project root:
  ```dotenv
  OPENAI_API_KEY="YOUR_ACTUAL_API_KEY"
  ```
  The integration tests load this file using `python-dotenv`. The SDK may also pick up keys directly from environment variables.
- **Agent Params:** YAML files define parameters like `name`, `instructions`, `model`, `goal`, etc.
- **Structured Output:** The `StudentAgent` is configured to return a JSON object matching the `StudentOutput` model defined in `student_expert_flow/models.py`.

Sample configurations are provided. You can modify these or create new ones.

## Usage

The core dialogue logic is implemented in `student_expert_flow/runner.py`. Currently, there is no direct command-line interface (CLI) entry point to run a dialogue easily. The main way to execute the dialogue is via the integration tests.

### Running Tests

**Standard Tests (Mocked):**

These tests verify the application logic without making real API calls.

```bash
poetry run pytest
```

**Integration Tests (Real API Calls):**

These tests run the actual dialogue flow using the configured language model, making real API calls.

**Requirements:**

- API Key configured in `.env` file or environment variables (see Configuration section).
- You can optionally control the number of turns the integration test runs using the `INTEGRATION_MAX_TURNS` environment variable (defaults to the value set in the test code, currently 2). Example: `export INTEGRATION_MAX_TURNS=3`
- These tests will incur API costs and take longer to run.

To run only the integration tests (shows dialogue output):

```bash
poetry run pytest -m integration -s
```

To run all tests _except_ integration tests:

```bash
poetry run pytest -m "not integration"
```
