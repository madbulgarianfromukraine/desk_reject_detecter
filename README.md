### README Update: Detection of Desk Reject by GenAI

This repository contains the implementation of a multi-agent system designed to automate the desk rejection process for scientific paper submissions (specifically targeting ICLR guidelines). It leverages Google Gemini models via Vertex AI to audit papers for various violations, including anonymity, formatting, safety, and scope.

### Key Features
*   **Multi-Agent Orchestration**: Specialized agents for Safety, Anonymity, Visual Integrity, Formatting, Policy, and Scope.
*   **Self-Correction**: Iterative execution logic that re-runs agents with low confidence scores to improve reliability.
*   **Multi-Modal Analysis**: Processes main PDF papers along with supplemental materials (images, logs, CSVs).
*   **Confidence Scoring**: Uses log-probabilities to calculate weighted confidence for each agent's findings.

---

### Prerequisites

#### 1. Requirements
Ensure you have Python 3 or higher installed. Install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `google-genai`: For interacting with Gemini models.
- `pydantic`: For structured data schemas and validation.
- `fire`: For the CLI interface.
- `scikit-learn` & `pandas`: For metric evaluation.

#### 2. Environment Variables
The system uses Google Vertex AI. You need to provide your credentials and project configuration. Create a file named `google.env` in the root directory:

```env
# Example google.env content
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

*(Note: The `core/config.py` is configured to load this file automatically.)*

---

### Basic Usage

The project provides a CLI via `main.py`.

#### Single Paper Analysis
To determine if a specific paper should be desk-rejected:

```bash
python main.py determine_desk_rejection --directory ./path/to/paper_folder
```
- **Optional Flags**:
  - `--think`: Enable Gemini's reasoning/thinking capabilities.
  - `--search`: Enable Google Search grounding for relevant agents.

#### Batch Evaluation
To evaluate a dataset of submissions and generate performance metrics (Precision, Recall, F1):

```bash
python main.py evaluate_desk_rejection --directory ./path/to/submissions_root --answers_only True --limit 10
```
- **Optional Flags**:
  - `--limit`: Limits the number of submissions to evaluate.
  - `--parallel`: Runs evaluations in parallel using a thread pool.
  - `--answers_only`: Set to `True` to evaluate only the precision of the final answer.
  - `--think`: Enable reasoning for agents.
  - `--search`: Enable search for agents.

---

### Project Structure
- `agents/`: Implementation of specialized auditor agents and their system prompts.
- `core/`: Core logic including LLM configuration (`config.py`), confidence calculation (`logprobs.py`), and data schemas (`schemas.py`).
- `ddr.py`: The main orchestrator that manages the multi-agent workflow and self-correction loop.
- `main.py`: CLI entry point for the system.
- `data/iclr/requirements/`: Contains the conference style guides and policy documents used as grounding for the agents.