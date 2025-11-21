from self_correcting_rag.crew import SelfCorrectingRag
from dotenv import load_dotenv
load_dotenv()


crew = SelfCorrectingRag()
def run(query: str):
    # inputs = {"query": query}
    # result=crew.crew().kickoff(inputs=inputs)
    result = crew.run_pipeline(query=user_query)

    return result

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    output = run(user_query)
    print("\n=== Final Output ===")
    print(output)
