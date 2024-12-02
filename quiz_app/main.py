from dotenv import load_dotenv
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
import wikipedia

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

### インプットからジャンルを理解する

class Genre(BaseModel):
    genre: str = Field(description="ユーザーがお題として欲しいクイズのジャンル")

genre_llm = llm.with_structured_output(Genre)

system = """あなたはユーザーのメッセージからどんなジャンルの問題を出題して欲しいのか理解します。\n。"""

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{message}")
    ]
)

genre_getter = input_prompt | genre_llm

### wikipediaのクエリを生成するチェーン

class WikipediaQuery(BaseModel):
    query: str = Field(description="wikipediaでの検索クエリ")

query_llm = llm.with_structured_output(WikipediaQuery)

system = """あなたはクイズの問題を作るための情報収集をします。\n
    wikipediaから出題のための情報を取得してくるため、適切な検索クエリを考えなさい。 \n
    ただし、ユーザーの求めているジャンルから乖離しすぎないように注意してください。"""

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{message}")
    ]
)

query_creator = input_prompt | query_llm

### クイズ生成するチェーン

class QuestionAndAnswer(BaseModel):
    question: str = Field(description="クイズの問題文")
    answer: str = Field(description="クイズの回答")

quiz_llm = llm.with_structured_output(QuestionAndAnswer)

system = """あなたはクイズの問題考案者です。与えられたドキュメントからそのジャンルに関するクイズを出題します。\n
    クイズのレベルはかなり高く知識がないと解けないようにしてください。内容はロジカルにステップバイステップで考えてロジカルに論理構造を理解して作りなさい。"""

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "ジャンル:{genre}, 取得されたドキュメント{document}")
    ]
)

quiz_generator = input_prompt | quiz_llm

### グラフの状態を定義するステート

class GraphState(MessagesState):
    genre: str
    query: str
    document: str
    question: str
    answer: str

### ジャンルを理解するnode
def get_genre(state):
    message = state["messages"][-1]

    response = genre_getter.invoke({"message": message})
    return {"genre": response.genre}

### クエリを生成するnode
def create_query(state):
    message = state["messages"][-1]

    response = query_creator.invoke({"message": message})
    return {"query": response.query}

### Wikipediaから情報を取得するnode
def search_wikipedia(state):
    query = state["query"]
    wikipedia.set_lang("ja")
    word = wikipedia.search(query, results=1)
    document = wikipedia.page(word).content

    return {"document": document}

### クイズを生成するnode
def quiz_generate(state):
    genre = state["genre"]
    document = state["document"]

    response = quiz_generator.invoke({"genre": genre,"document": document})
    return {"messages": response.question, "question": response.question, "answer": response.answer}

    
builder = StateGraph(GraphState)

builder.add_node("get_genre", get_genre)
builder.add_node("create_query", create_query)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("quiz_generate", quiz_generate)

builder.add_edge(START, "get_genre")
builder.add_edge("get_genre", "create_query")
builder.add_edge("create_query", "search_wikipedia")
builder.add_edge("search_wikipedia", "quiz_generate")
builder.add_edge("quiz_generate", END)

graph = builder.compile()

app = FastAPI()

class InputRequest(BaseModel):
    user_input: str

@app.post("/llm")
async def invoke(request: InputRequest):
    result = graph.invoke({"messages": request.user_input})
    return {"responce": result["messages"][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")




