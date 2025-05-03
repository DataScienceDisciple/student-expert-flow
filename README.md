# Student-Expert Flow CLI

This tool facilitates dialogues between a 'student' AI agent learning a topic and an 'expert' AI agent providing information, leveraging the `openai-agents-python` library. The student agent typically has a learning goal, and the conversation continues until the goal is met or a maximum number of turns is reached.

## Features

- Run AI-driven dialogues based on configurable agent roles and goals.
- Utilizes OpenAI models via the `openai-agents-python` SDK.
- Supports structured JSON output from the student agent to assess goal completion.
- Integrates web search for the expert agent to fetch up-to-date information.
- Saves full conversation transcripts in Markdown format (`.md`) and generates concise summaries.
- Configurable via YAML files for agent parameters (instructions, model, goal, etc.).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd student-expert-flow
    ```
2.  **Install Poetry:**
    If you don't have Poetry, follow the installation instructions [here](https://python-poetry.org/docs/#installation).
3.  **Install Dependencies:**
    ```bash
    poetry install
    ```
    This command also installs the `student-expert-flow` CLI tool within the virtual environment.
4.  **Configure API Key:**
    Create a `.env` file in the project root and add your OpenAI API key:
    ```dotenv
    # .env
    OPENAI_API_KEY="YOUR_ACTUAL_API_KEY"
    ```
5.  **Activate Environment:**
    ```bash
    poetry shell
    ```

## Configuration

- **Agents:** Agent behavior (name, instructions, model, goal, tools) is defined in YAML files within the `configs/` directory. Modify existing examples or create new ones.
- **API Key:** Loaded from the `.env` file (or environment variables).

## Usage

Once the setup is complete and the environment is activated (`poetry shell`), you can run dialogues using the `student-expert-flow` command.

**Basic Command:**

```bash
student-expert-flow --student-config <path_to_student.yaml> --expert-config <path_to_expert.yaml>
```

**Arguments:**

- `--student-config` (Required): Path to the Student agent's YAML configuration file (e.g., `configs/student_config.yaml`).
- `--expert-config` (Required): Path to the Expert agent's YAML configuration file (e.g., `configs/expert_config.yaml`).
- `--max-turns` (Optional): Maximum number of dialogue turns. Defaults to 5.
- `--output-dir` (Optional): Directory to save conversation transcripts (as `.md`) and summaries (as `.txt`). Defaults to `transcripts/`.

**Example:**

Run a dialogue using the newsletter agent configurations, limit to 3 turns, and save output to `results/`:

```bash
student-expert-flow --student-config configs/student_newsletter_config.yaml --expert-config configs/expert_newsletter_config.yaml --max-turns 3 --output-dir results
```

The dialogue will run in your terminal, and the transcript/summary files will be saved to the specified output directory upon completion.
