from dotenv import load_dotenv
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def assistant(state: MessagesState):
    return {"messages": llm.invoke(state["messages"][-1].content)}

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)

builder.add_edge(START, "assistant")

graph = builder.compile()

app = FastAPI()

class InputRequest(BaseModel):
    user_input: str

@app.post("/llm")
async def invoke(request: InputRequest):
    result = graph.invoke({"messages": request.user_input})
    return {"responce": result["messages"][-1].content}


@app.get("/")
async def root():
    return {"message", "Hello, World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")




