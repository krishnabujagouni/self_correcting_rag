Self-Correcting RAG with CrewAI

A robust, multi-agent Retrieval-Augmented Generation (RAG) system built using CrewAI. This system goes beyond simple retrieval by implementing a self-correcting loop that evaluates answers, filters noise, and actively reformulates search queries when information is missing.

🚀 Key Features

Self-Correction Loop: If the generated answer is poor, the system analyzes why and either revises the answer or reformulates the search query to find better data.

Hallucination Guardrails: An explicit Guardrail Agent filters retrieved context to remove irrelevant noise before it reaches the generator.

Quality Evaluation: An Evaluator Agent scores every answer on a 0-100 scale. Low scores trigger a correction workflow.

Loop Prevention: Built-in logic prevents infinite loops by setting strict maximum retry limits (default: 3 iterations).

Context Deduplication: Custom logic ensures the LLM context window isn't flooded with repetitive retrieved text.

🤖 Agents & Architecture

The system utilizes a crew of specialized AI agents:

Retriever Agent: Fetches raw documents from the knowledge base.

Guardrail Agent: Acts as a strict editor, removing irrelevant information from the retrieved context.

Generator Agent: Synthesizes a concise, factual answer based only on the filtered context.

Evaluator Agent: Grades the answer for accuracy and completeness. It decides the next step: pass, revise, or retrieve_more_data.

ReGenerator Agent: Rewrites answers to address specific concerns raised by the evaluator.

Query Reformulator: Creates optimized search queries if the original retrieval failed to yield useful results.

🛠️ Installation

Clone the repository:

git clone [https://github.com/yourusername/self-correcting-rag.git](https://github.com/yourusername/self-correcting-rag.git)
cd self-correcting-rag


Install dependencies:
This project requires Python 3.10+ and CrewAI.

pip install crewai crewai-tools


Set up Environment Variables:
You need a Google Gemini API key (or OpenAI key if configured).

Windows (PowerShell):

$env:GEMINI_API_KEY="your_api_key_here"


Mac/Linux:

export GEMINI_API_KEY="your_api_key_here"


⚙️ Configuration

The agent behaviors and task definitions are separated into YAML files for easy editing.

config/agents.yaml: Defines the prompts, backstories, and specific rules for each agent (e.g., "Do not add conversational filler").

config/tasks.yaml: Defines the specific outputs expected from each task (e.g., "Return ONLY valid JSON").

🏃‍♂️ Usage

Run the main script to start the pipeline. You can modify the query inside the __main__ block of self_correcting_rag.py.

python self_correcting_rag.py


Example Output

🚀 SELF-CORRECTING RAG PIPELINE
======================================================================
📝 Query: What is agentic AI?
🔄 Max loops: 3
🛡️ Guardrail: Enabled
======================================================================

... [Agents working] ...

📊 Quality Score: 95/100
📌 Decision: PASS

🤖 FINAL ANSWER:
Agentic AI refers to artificial intelligence systems that can operate autonomously 
to achieve specific goals. These agents can perceive their environment, make 
decisions, and take actions without continuous human intervention.


🔧 Customization

Adjusting "Strictness"

If the Evaluator is too harsh or too lenient, edit the CRITICAL SCORING RULES in config/agents.yaml.

Disabling Guardrails

For faster execution on simple queries, you can disable the guardrail in self_correcting_rag.py:

result = crew.run_pipeline(
    query="Your query",
    max_loops=3,
    use_guardrail=False  # Set to False to skip the filtering step
)


🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
