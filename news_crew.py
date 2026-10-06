from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()
llm = LLM(model="gpt-4o-mini")

search = SerperDevTool()

researcher = Agent(
    role="Researcher",
    goal="Finds today's top 3 tech news stories", 
    backstory="You search the internet for fresh facts.",
    tools=[search],
    verbose=True
)
writer = Agent(
    role="Content Writer", 
    goal="Write a newsletter fpr each tech news with 5-line of content", 
    backstory="You are well experience tech content write and used to write newsletters from the facts", 
    verbose=True
)

find = Task(
    description="Search the web and tell me top 3 recent tech news stories.",
    expected_output="Top 3 recent tech news headline with a short note.",
    agent=researcher
)

write = Task(
    description="Use the top 3 tech news to write a newsletter with 5-sentence for each tech news article.",
    expected_output="Draft the newsletter clearly and with summarization.", 
    agent=writer,
    context=[find]
)

crew = Crew(
    agents=[researcher, writer], 
    tasks=[find, write],
    process=Process.sequential, 
    verbose=True
)

print(crew.kickoff())