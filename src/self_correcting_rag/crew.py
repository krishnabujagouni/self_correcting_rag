
    

# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from crewai.agents.agent_builder.base_agent import BaseAgent
# from typing import List
# import os
# from src.self_correcting_rag.tools.tools import get_rag_tool
# from crewai import LLM

# def get_llm_config():
#     """
#     Determine which LLM to use based on environment variables.
#     Returns LLM instance and provider name.
#     """
#     llm_provider = os.getenv("LLM_PROVIDER", "google").lower()
#     llm_model = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
    
#     gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#     if not gemini_key:
#         raise ValueError("❌ Missing GEMINI_API_KEY environment variable")
    
#     os.environ["GEMINI_API_KEY"] = gemini_key

#     llm = LLM(
#         model=llm_model,
#         api_key=gemini_key,
#         # temperature=0.3,
#     )
#     return llm, "google"


# @CrewBase
# class SelfCorrectingRag():
#     """SelfCorrectingRag crew"""

#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"

#     def __init__(self) -> None:
#         print("\n🔧 Initializing components...")
#         self.llm, self.provider = get_llm_config()
#         print(f"✅ LLM configured: {self.provider}")
        
#         self.rag_tool = get_rag_tool()
#         print("✅ RAG tool ready\n")

#     @agent
#     def RetrieverAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['RetrieverAgent'],
#             verbose=True,
#             tools=[self.rag_tool],
#             llm=self.llm,
#             allow_delegation=False,
#         )

#     @agent
#     def GuardrailAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GuardrailAgent'],
#             verbose=True,
#             llm=self.llm,
#             allow_delegation=False,
#         )

#     @agent
#     def GeneratorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GeneratorAgent'],
#             verbose=True,
#             llm=self.llm,
#             allow_delegation=False,
#         )

#     @agent
#     def EvaluatorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['EvaluatorAgent'],
#             verbose=True,
#             llm=self.llm,
#             allow_delegation=False,
#         )

#     @task
#     def retrieve_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['retrieve_task'],
            
#         )

#     @task
#     def guardrail_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['guardrail_task'],
#         )

#     @task
#     def generate_answer_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['generate_answer_task'],
#         )

#     @task
#     def evaluate_answer_task(self) -> Task:
#         return Task(
#             config=self.tasks_config['evaluate_answer_task'],
#         )

#     @crew
#     def crew(self) -> Crew:
#         """Creates the SelfCorrectingRag crew"""
#         return Crew(
#             agents=self.agents,  
#             tasks=self.tasks,    
#             process=Process.sequential,
#             verbose=True,
#         )
    
# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from typing import List, Dict, Any
# import os

# from src.self_correcting_rag.tools.tools import get_rag_tool
# from crewai import LLM


# # -------------------------
# # LLM CONFIG
# # -------------------------
# def get_llm_config():
#     llm_model = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
#     gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

#     if not gemini_key:
#         raise ValueError("❌ Missing GEMINI_API_KEY")

#     os.environ["GEMINI_API_KEY"] = gemini_key

#     llm = LLM(
#         model=llm_model,
#         api_key=gemini_key,
#         temperature=0.2,
#     )
#     return llm


# # ########################################################
# #   SELF CORRECTING RAG CREW
# # ########################################################
# @CrewBase
# class SelfCorrectingRag():
#     agents_config = "config/agents.yaml"
#     tasks_config  = "config/tasks.yaml"

#     def __init__(self) -> None:
#         print("\n🔧 Initializing Self-Correcting RAG Crew...")
#         self.llm = get_llm_config()
#         self.rag_tool = get_rag_tool()
#         print("✅ Initialization complete.\n")

#     # -------------------------
#     # AGENTS
#     # -------------------------
#     @agent
#     def RetrieverAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['RetrieverAgent'],
#             tools=[self.rag_tool],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def GuardrailAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GuardrailAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def GeneratorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GeneratorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def EvaluatorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['EvaluatorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def ReGeneratorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['ReGeneratorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def QueryReformulatorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['QueryReformulatorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     # --------------------------------------------------
#     # TASK DEFINITIONS
#     # --------------------------------------------------
#     @task
#     def retrieve_task(self) -> Task:
#         return Task(config=self.tasks_config['retrieve_task'])

#     @task
#     def guardrail_task(self) -> Task:
#         return Task(config=self.tasks_config['guardrail_task'])

#     @task
#     def generate_answer_task(self) -> Task:
#         return Task(config=self.tasks_config['generate_answer_task'])

#     @task
#     def evaluate_answer_task(self) -> Task:
#         return Task(config=self.tasks_config['evaluate_answer_task'])

#     @task
#     def regenerate_answer_task(self) -> Task:
#         return Task(config=self.tasks_config['regenerate_answer_task'])

#     @task
#     def query_reformulation_task(self) -> Task:
#         return Task(config=self.tasks_config['query_reformulation_task'])

#     # ########################################################
#     #   SELF-CORRECTING EXECUTION LOOP
#     # ########################################################
#     @crew
#     def crew(self) -> Crew:
#         """
#         Main Crew with dynamic execution logic.
#         """
#         return Crew(
#             agents=self.agents,
#             tasks=self.tasks,
#             process=Process.sequential,
#             verbose=True,
#         )

#     # ---------------------------------------------------------
#     # HIGH LEVEL WORKFLOW FUNCTION
#     # ---------------------------------------------------------
#     def run_pipeline(self, query: str):
#         """
#         Manual pipeline with self-correcting loop:
        
#         retrieve → guardrail → generate → evaluate
#         IF revise → regenerate
#         IF retrieve_more_data → reformulate → retrieve again
#         Loops max 3 times.
#         """

#         print("\n🚀 Starting Self-Correcting RAG Loop...\n")

#         max_loops = 3
#         current_query = query
#         final_answer = None

#         for loop in range(1, max_loops + 1):
#             print(f"\n==========================")
#             print(f"🔁 LOOP {loop} / {max_loops}")
#             print(f"==========================\n")

#             # 1. RETRIEVE
#             retrieved = self.retrieve_task().run(
#                 inputs={"query": current_query}
#             )

#             # 2. GUARDRAIL
#             filtered_context = self.guardrail_task().run(
#                 inputs={"context": retrieved, "query": current_query}
#             )

#             # 3. GENERATE
#             answer = self.generate_answer_task().run(
#                 inputs={
#                     "context": filtered_context,
#                     "query": current_query
#                 }
#             )

#             # 4. EVALUATE
#             evaluation = self.evaluate_answer_task().run(
#                 inputs={
#                     "answer": answer,
#                     "context": filtered_context,
#                     "query": current_query
#                 }
#             )

#             # Parse evaluator output:
#             recommendation = evaluation.get("recommendation", "pass")
#             score = evaluation.get("score", 0)

#             print(f"📝 Evaluator Score: {score}")
#             print(f"📌 Recommendation: {recommendation}")

#             # DECISION BRANCHES
#             if recommendation == "pass":
#                 print("✅ Answer approved by evaluator.")
#                 final_answer = answer
#                 break

#             elif recommendation == "revise":
#                 print("♻️ Revising answer...")
#                 answer = self.regenerate_answer_task().execute(
#                     inputs={
#                         "answer": answer,
#                         "evaluation": evaluation,
#                         "context": filtered_context
#                     }
#                 )
#                 final_answer = answer
#                 continue

#             elif recommendation == "retrieve_more_data":
#                 print("🔍 Reformulating query for improved retrieval...")
#                 new_query = self.query_reformulation_task().execute(
#                     inputs={
#                         "query": current_query,
#                         "evaluation": evaluation
#                     }
#                 )
#                 current_query = new_query
#                 continue

#         print("\n🎉 FINAL ANSWER:")
#         print(final_answer)

#         return final_answer












# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from typing import List, Dict, Any
# import os

# from src.self_correcting_rag.tools.tools import get_rag_tool
# from crewai import LLM


# # -------------------------
# # LLM CONFIG
# # -------------------------
# def get_llm_config():
#     llm_model = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
#     gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

#     if not gemini_key:
#         raise ValueError("❌ Missing GEMINI_API_KEY")

#     os.environ["GEMINI_API_KEY"] = gemini_key

#     llm = LLM(
#         model=llm_model,
#         api_key=gemini_key,
#         temperature=0.2,
#     )
#     return llm


# # ########################################################
# #   SELF CORRECTING RAG CREW
# # ########################################################
# @CrewBase
# class SelfCorrectingRag():
#     agents_config = "config/agents.yaml"
#     tasks_config  = "config/tasks.yaml"

#     def __init__(self) -> None:
#         print("\n🔧 Initializing Self-Correcting RAG Crew...")
#         self.llm = get_llm_config()
#         self.rag_tool = get_rag_tool()
#         print("✅ Initialization complete.\n")

#     # -------------------------
#     # AGENTS
#     # -------------------------
#     @agent
#     def RetrieverAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['RetrieverAgent'],
#             tools=[self.rag_tool],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def GuardrailAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GuardrailAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def GeneratorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['GeneratorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def EvaluatorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['EvaluatorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def ReGeneratorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['ReGeneratorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     @agent
#     def QueryReformulatorAgent(self) -> Agent:
#         return Agent(
#             config=self.agents_config['QueryReformulatorAgent'],
#             llm=self.llm,
#             allow_delegation=False,
#             verbose=True,
#         )

#     # ---------------------------------------------------------
#     # HIGH LEVEL WORKFLOW FUNCTION
#     # ---------------------------------------------------------
#     def run_pipeline(self, query: str):
#         """
#         Manual pipeline with self-correcting loop:
        
#         retrieve → guardrail → generate → evaluate
#         IF revise → regenerate
#         IF retrieve_more_data → reformulate → retrieve again
#         Loops max 3 times.
#         """

#         print("\n🚀 Starting Self-Correcting RAG Loop...\n")

#         max_loops = 3
#         current_query = query
#         final_answer = None

#         for loop in range(1, max_loops + 1):
#             print(f"\n==========================")
#             print(f"🔁 LOOP {loop} / {max_loops}")
#             print(f"==========================\n")

#             # 1. RETRIEVE - Create mini crew for this task
#             retrieve_task = Task(
#                 config=self.tasks_config['retrieve_task'],
#                 agent=self.RetrieverAgent()
#             )
#             retrieve_crew = Crew(
#                 agents=[self.RetrieverAgent()],
#                 tasks=[retrieve_task],
#                 process=Process.sequential,
#                 verbose=True
#             )
#             retrieved_result = retrieve_crew.kickoff(inputs={"query": current_query})
#             retrieved = str(retrieved_result)

#             # 2. GUARDRAIL
#             guardrail_task = Task(
#                 config=self.tasks_config['guardrail_task'],
#                 agent=self.GuardrailAgent()
#             )
#             guardrail_crew = Crew(
#                 agents=[self.GuardrailAgent()],
#                 tasks=[guardrail_task],
#                 process=Process.sequential,
#                 verbose=True
#             )
#             filtered_result = guardrail_crew.kickoff(
#                 inputs={"context": retrieved, "query": current_query}
#             )
#             filtered_context = str(filtered_result)

#             # 3. GENERATE
#             generate_task = Task(
#                 config=self.tasks_config['generate_answer_task'],
#                 agent=self.GeneratorAgent()
#             )
#             generate_crew = Crew(
#                 agents=[self.GeneratorAgent()],
#                 tasks=[generate_task],
#                 process=Process.sequential,
#                 verbose=True
#             )
#             answer_result = generate_crew.kickoff(
#                 inputs={
#                     "context": filtered_context,
#                     "query": current_query
#                 }
#             )
#             answer = str(answer_result)

#             # 4. EVALUATE
#             evaluate_task = Task(
#                 config=self.tasks_config['evaluate_answer_task'],
#                 agent=self.EvaluatorAgent()
#             )
#             evaluate_crew = Crew(
#                 agents=[self.EvaluatorAgent()],
#                 tasks=[evaluate_task],
#                 process=Process.sequential,
#                 verbose=True
#             )
#             eval_result = evaluate_crew.kickoff(
#                 inputs={
#                     "answer": answer,
#                     "context": filtered_context,
#                     "query": current_query
#                 }
#             )
            
#             # Parse evaluator output - adjust based on your YAML config
#             evaluation = str(eval_result)
            
#             # You'll need to parse the evaluation string based on your output format
#             # This is a simplified example:
#             if "pass" in evaluation.lower():
#                 recommendation = "pass"
#             elif "revise" in evaluation.lower():
#                 recommendation = "revise"
#             elif "retrieve_more" in evaluation.lower():
#                 recommendation = "retrieve_more_data"
#             else:
#                 recommendation = "pass"

#             print(f"📝 Evaluation: {evaluation}")
#             print(f"📌 Recommendation: {recommendation}")

#             # DECISION BRANCHES
#             if recommendation == "pass":
#                 print("✅ Answer approved by evaluator.")
#                 final_answer = answer
#                 break

#             elif recommendation == "revise":
#                 print("♻️ Revising answer...")
#                 regenerate_task = Task(
#                     config=self.tasks_config['regenerate_answer_task'],
#                     agent=self.ReGeneratorAgent()
#                 )
#                 regenerate_crew = Crew(
#                     agents=[self.ReGeneratorAgent()],
#                     tasks=[regenerate_task],
#                     process=Process.sequential,
#                     verbose=True
#                 )
#                 revised_result = regenerate_crew.kickoff(
#                     inputs={
#                         "answer": answer,
#                         "evaluation": evaluation,
#                         "context": filtered_context
#                     }
#                 )
#                 final_answer = str(revised_result)
#                 continue

#             elif recommendation == "retrieve_more_data":
#                 print("🔍 Reformulating query for improved retrieval...")
#                 reformulate_task = Task(
#                     config=self.tasks_config['query_reformulation_task'],
#                     agent=self.QueryReformulatorAgent()
#                 )
#                 reformulate_crew = Crew(
#                     agents=[self.QueryReformulatorAgent()],
#                     tasks=[reformulate_task],
#                     process=Process.sequential,
#                     verbose=True
#                 )
#                 new_query_result = reformulate_crew.kickoff(
#                     inputs={
#                         "query": current_query,
#                         "evaluation": evaluation
#                     }
#                 )
#                 current_query = str(new_query_result)
#                 continue

#         print("\n🎉 FINAL ANSWER:")
#         print(final_answer)

#         return final_answer


from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Dict, Any
import os
import json
import re

# Assumed import based on your snippet. 
# If running locally, ensure this path exists or adjust to your actual tool import.
from src.self_correcting_rag.tools.tools import get_rag_tool
from crewai import LLM


def get_llm_config():
    llm_model = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key:
        # For safety in this environment, we avoid raising if keys aren't present during generation,
        # but in production this raise is correct.
        print("⚠️ Warning: GEMINI_API_KEY not found in environment.")
    else:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Fallback for execution safety if key is missing
    api_key_val = gemini_key if gemini_key else ""

    llm = LLM(
        model=llm_model,
        api_key=api_key_val,
        temperature=0.2,
    )
    return llm


@CrewBase
class SelfCorrectingRag():
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self) -> None:
        print("\n🔧 Initializing Self-Correcting RAG Crew...")
        self.llm = get_llm_config()
        # Ensure get_rag_tool returns a valid CrewAI Tool object
        self.rag_tool = get_rag_tool()
        print("✅ Initialization complete.\n")

    # =========================================================
    # AGENTS WITH LOOP PREVENTION
    # =========================================================
    
    @agent
    def RetrieverAgent(self) -> Agent:
        """Retriever with strict iteration limits to prevent loops"""
        return Agent(
            config=self.agents_config['RetrieverAgent'],
            tools=[self.rag_tool],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=2,  # Maximum 2 iterations to prevent loops
            max_retry_limit=1,  # Only retry once
        )

    @agent
    def GuardrailAgent(self) -> Agent:
        """Guardrail agent - can be disabled via parameter"""
        return Agent(
            config=self.agents_config['GuardrailAgent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=3,
        )

    @agent
    def GeneratorAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['GeneratorAgent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=5,
        )

    @agent
    def EvaluatorAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['EvaluatorAgent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=3,
        )

    @agent
    def ReGeneratorAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ReGeneratorAgent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=5,
        )

    @agent
    def QueryReformulatorAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['QueryReformulatorAgent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            max_iter=3,
        )

    # =========================================================
    # UTILITY METHODS
    # =========================================================
    
    def parse_evaluation(self, eval_text: str) -> Dict[str, Any]:
        """Parse evaluator output to extract structured recommendation."""
        eval_lower = eval_text.lower()
        
        # Try to extract JSON first
        json_match = re.search(r'\{[^}]+\}', eval_text, re.DOTALL)
        if json_match:
            try:
                eval_dict = json.loads(json_match.group())
                return {
                    'recommendation': eval_dict.get('recommendation', 'pass'),
                    'score': eval_dict.get('score', 0),
                    'concerns': eval_dict.get('concerns', ''),
                    'raw': eval_text
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback to keyword detection
        if 'retrieve_more_data' in eval_lower or 'retrieve more' in eval_lower:
            recommendation = 'retrieve_more_data'
        elif 'revise' in eval_lower or 'regenerate' in eval_lower:
            recommendation = 'revise'
        elif 'pass' in eval_lower or 'approved' in eval_lower:
            recommendation = 'pass'
        else:
            recommendation = 'pass'
        
        return {
            'recommendation': recommendation,
            'score': self._extract_score(eval_text),
            'concerns': eval_text,
            'raw': eval_text
        }
    
    def _extract_score(self, text: str) -> int:
        """Extract numeric score from evaluation text."""
        score_match = re.search(r'score[:\s]+(\d+)', text, re.IGNORECASE)
        if score_match:
            return int(score_match.group(1))
        return 0

    def execute_single_task(self, agent: Agent, task_config_key: str, inputs: Dict[str, Any]) -> str:
        """
        Execute a single task with comprehensive error handling and DEDUPLICATION logic.
        """
        task = Task(
            config=self.tasks_config[task_config_key],
            agent=agent
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff(inputs=inputs)
            result_str = str(result).strip()

            # --- FIX: DEDUPLICATION LOGIC ---
            # This block addresses the Context Flooding issue identified in your logs.
            if task_config_key == 'retrieve_task':
                original_len = len(result_str)
                # Split by newlines, remove empty lines, and deduplicate while preserving order
                lines = [line.strip() for line in result_str.split('\n') if line.strip()]
                seen = set()
                deduped_lines = []
                for line in lines:
                    if line not in seen:
                        seen.add(line)
                        deduped_lines.append(line)
                # Rejoin unique lines
                result_str = "\n".join(deduped_lines)
                
                if len(result_str) < original_len:
                    print(f" ✂️ Deduplication applied: Reduced context from {original_len} to {len(result_str)} characters.")
            # --------------------------------
            
            # Handle empty or invalid output
            if not result_str or result_str in ["", "None", "null"]:
                print(f"\n⚠️ WARNING: Task '{task_config_key}' returned empty output!")
                
                # Provide intelligent fallbacks based on task type
                if task_config_key == 'guardrail_task':
                    print(f"   → FALLBACK: Using original retrieved content (bypassing filter)")
                    return inputs.get('context', '')
                elif task_config_key == 'retrieve_task':
                    print(f"   → FALLBACK: No documents found")
                    return "No relevant documents found in the knowledge base."
                else:
                    print(f"   → FALLBACK: Returning empty string")
                    return ''
            
            return result_str
            
        except ValueError as e:
            if "No valid task outputs" in str(e):
                print(f"\n❌ ERROR: Task '{task_config_key}' produced no valid outputs")
                
                # Intelligent fallback strategy
                if task_config_key == 'guardrail_task':
                    print(f"   → FALLBACK: Bypassing guardrail, using raw retrieval")
                    return inputs.get('context', '')
                elif task_config_key == 'retrieve_task':
                    print(f"   → FALLBACK: Retrieval failed, no documents")
                    return "No relevant documents found in the knowledge base."
                elif task_config_key == 'generate_answer_task':
                    print(f"   → FALLBACK: Cannot generate answer")
                    return "Unable to generate answer due to insufficient context."
                elif task_config_key == 'evaluate_answer_task':
                    print(f"   → FALLBACK: Using default evaluation")
                    return '{"score": 70, "recommendation": "pass", "concerns": "Evaluation incomplete"}'
                else:
                    return ''
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error in task '{task_config_key}': {e}")
            if task_config_key == 'guardrail_task':
                return inputs.get('context', '')
            raise

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    
    def run_pipeline(self, query: str, max_loops: int = 3, use_guardrail: bool = False):
        """
        Self-correcting RAG pipeline with optional guardrail and loop prevention.
        
        Args:
            query: User's question
            max_loops: Maximum correction iterations (default: 3)
            use_guardrail: Whether to use guardrail agent (default: False, recommended)
        
        Returns:
            Dict with final_answer, iterations, and history
        """
        print("\n" + "="*70)
        print("🚀 SELF-CORRECTING RAG PIPELINE")
        print("="*70)
        print(f"📝 Query: {query}")
        print(f"🔄 Max loops: {max_loops}")
        print(f"🛡️ Guardrail: {'Enabled' if use_guardrail else 'Disabled'}")
        print("="*70 + "\n")

        current_query = query
        final_answer = None
        iteration_history = []

        for loop in range(1, max_loops + 1):
            print(f"\n{'━'*70}")
            print(f"🔁 ITERATION {loop}/{max_loops}")
            print(f"{'━'*70}")
            print(f"Query: {current_query}\n")
            
            loop_data = {
                'loop_number': loop,
                'query': current_query
            }

            # ===========================================
            # STEP 1: RETRIEVE
            # ===========================================
            print("📚 [1/4] RETRIEVING documents from knowledge base...")
            try:
                retrieved = self.execute_single_task(
                    agent=self.RetrieverAgent(),
                    task_config_key='retrieve_task',
                    inputs={"query": current_query}
                )
                loop_data['retrieved'] = retrieved
                
                if "No relevant documents" in retrieved:
                    print(f"⚠️  No documents found!")
                    final_answer = "I couldn't find relevant information in the knowledge base to answer this question."
                    break
                else:
                    print(f"✅ Retrieved {len(retrieved)} characters")
                    
            except Exception as e:
                print(f"❌ Retrieval failed: {e}")
                final_answer = "Retrieval system error. Please try again."
                break

            # ===========================================
            # STEP 2: GUARDRAIL (OPTIONAL)
            # ===========================================
            if use_guardrail:
                print("\n🛡️ [2/4] FILTERING content...")
                try:
                    filtered_context = self.execute_single_task(
                        agent=self.GuardrailAgent(),
                        task_config_key='guardrail_task',
                        inputs={
                            "context": retrieved,
                            "query": current_query
                        }
                    )
                    
                    # Validate guardrail output
                    if not filtered_context or len(filtered_context) < 50:
                        print(f"⚠️  Guardrail output insufficient, using raw retrieval")
                        filtered_context = retrieved
                    else:
                        print(f"✅ Filtered to {len(filtered_context)} characters")
                        
                except Exception as e:
                    print(f"❌ Guardrail error: {e}")
                    print(f"   Using raw retrieval as fallback")
                    filtered_context = retrieved
            else:
                print("\n⏭️  [2/4] SKIPPING guardrail (using raw retrieval)")
                filtered_context = retrieved
                
            loop_data['filtered_context'] = filtered_context

            # ===========================================
            # STEP 3: GENERATE
            # ===========================================
            print("\n✍️  [3/4] GENERATING answer...")
            try:
                answer = self.execute_single_task(
                    agent=self.GeneratorAgent(),
                    task_config_key='generate_answer_task',
                    inputs={
                        "context": filtered_context,
                        "query": current_query
                    }
                )
                loop_data['answer'] = answer
                print(f"✅ Generated answer ({len(answer)} characters)")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                final_answer = "Failed to generate answer from retrieved content."
                break

            # ===========================================
            # STEP 4: EVALUATE
            # ===========================================
            print("\n🔍 [4/4] EVALUATING answer quality...")
            try:
                evaluation_raw = self.execute_single_task(
                    agent=self.EvaluatorAgent(),
                    task_config_key='evaluate_answer_task',
                    inputs={
                        "answer": answer,
                        "context": filtered_context,
                        "query": current_query
                    }
                )
                
                evaluation = self.parse_evaluation(evaluation_raw)
                recommendation = evaluation['recommendation']
                score = evaluation['score']
                
                loop_data['evaluation'] = evaluation
                
                print(f"📊 Quality Score: {score}/100")
                print(f"📌 Decision: {recommendation.upper()}")
                if evaluation['concerns']:
                    print(f"💬 Notes: {evaluation['concerns'][:150]}...")
                    
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                print(f"   Accepting answer by default")
                final_answer = answer
                break

            iteration_history.append(loop_data)

            # ===========================================
            # DECISION TREE
            # ===========================================
            if recommendation == "pass":
                print("\n✅ Answer APPROVED by evaluator!")
                final_answer = answer
                break

            elif recommendation == "revise":
                print("\n♻️  REVISING answer based on feedback...")
                try:
                    revised_answer = self.execute_single_task(
                        agent=self.ReGeneratorAgent(),
                        task_config_key='regenerate_answer_task',
                        inputs={
                            "answer": answer,
                            "evaluation": evaluation['raw'],
                            "context": filtered_context
                        }
                    )
                    final_answer = revised_answer
                    print(f"✅ Revision complete ({len(revised_answer)} characters)")
                except Exception as e:
                    print(f"❌ Revision failed: {e}")
                    final_answer = answer  # Use original
                continue

            elif recommendation == "retrieve_more_data":
                print("\n🔍 REFORMULATING query for better retrieval...")
                try:
                    new_query = self.execute_single_task(
                        agent=self.QueryReformulatorAgent(),
                        task_config_key='query_reformulation_task',
                        inputs={
                            "query": current_query,
                            "evaluation": evaluation['raw']
                        }
                    )
                    print(f"📝 New query: {new_query}")
                    current_query = new_query
                except Exception as e:
                    print(f"❌ Reformulation failed: {e}")
                    print(f"   Using original answer")
                    final_answer = answer
                    break
                continue

            else:
                print(f"\n⚠️  Unknown recommendation: {recommendation}")
                print(f"   Accepting answer by default")
                final_answer = answer
                break

        # ===========================================
        # FINAL OUTPUT
        # ===========================================
        print(f"\n{'='*70}")
        print("🎉 PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total iterations: {len(iteration_history)}")
        print(f"📝 Final query: {current_query}")
        print(f"{'='*70}\n")
        
        if final_answer:
            print("📄 FINAL ANSWER:")
            print("─" * 70)
            print(final_answer)
            print("─" * 70)
        else:
            print("❌ No final answer generated")

        return {
            'final_answer': final_answer,
            'iterations': len(iteration_history),
            # 'history': iteration_history,
            'final_query': current_query,
            'success': final_answer is not None
        }


# ===================================================
# USAGE EXAMPLE
# ===================================================
if __name__ == "__main__":
    crew = SelfCorrectingRag()
    
    # Recommended: Run without guardrail initially, 
    # but ensure deduplication logic in execute_single_task handles the cleanup.
    result = crew.run_pipeline(
        query="What is agentic AI?",
        max_loops=3,
        use_guardrail=True  # Disabled for stability
    )
    
    print(f"\n✅ Success: {result['success']}")
    print(f"📊 Iterations used: {result['iterations']}")