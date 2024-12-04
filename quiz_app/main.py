import os
from contextlib import asynccontextmanager

import wikipedia
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

load_dotenv()
USER_NAME = os.environ["USER_NAME"]
USER_PASS = os.environ["USER_PASS"]

llm = ChatOpenAI(model="gpt-4o")


### インプットがクイズを生成して欲しいか答え合わせなのか判断するチェーン
class InputType(BaseModel):
    input_type: str = Field(
        description="ユーザーのメッセージがクイズの出題以来なら'question'、答え合わせの時は'answer'、ヒントを欲しがっている時は'hint'"
    )


input_type_llm = llm.with_structured_output(InputType)

system = """ユーザーからのメッセージの内容と意図を理解して、クイズの出題依頼だったら'question'、答え合わせなら'answer'、ヒントを欲しがっているときは'hint'と答えなさい。\n
        ただし、メッセージ履歴を見てまだ一回も問題を生成していない場合は出題依頼と解釈してください。
        """

input_prompt = ChatPromptTemplate.from_messages(
    [("system", system), MessagesPlaceholder("messages")]
)

input_router = input_prompt | input_type_llm

### 答え合わせをするチェーン
system = """あなたはクイズの出題者としてユーザーの回答に対してフィードバックをした上で正しい答えを解説付きで教えてあげてください\n
        ドキュメント:{document}\n\n
        クイズの問題:{question}\n正しい答え:{answer}"""

input_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("user", "{message}")]
)

answer_teller = input_prompt | llm | StrOutputParser()

### ヒントを生成するチェーン
system = """あなたはクイズの出題者として、回答者のヒントへの要求に答えてください。ただし、直接回答を教えてはいけません。\n
        過去のメッセージ履歴を見て、ヒントが被らないようにしてください。複数回ヒントを出すことを想定して少しずつ回答に近づけるようなヒントを考えてください。
        ドキュメント:{document}\n\n
        クイズの問題:{question}\n正しい答え:{answer}"""

input_prompt = ChatPromptTemplate.from_messages(
    [("system", system), MessagesPlaceholder("messages")]
)

hint_generator = input_prompt | llm | StrOutputParser()

### インプットからジャンルを理解する


class Genre(BaseModel):
    genre: str = Field(description="ユーザーがお題として欲しいクイズのジャンル")


genre_llm = llm.with_structured_output(Genre)

system = """あなたはユーザーのメッセージからどんなジャンルの問題を出題して欲しいのか理解します。\n。"""

input_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{message}")]
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
    [("system", system), ("human", "{message}")]
)

query_creator = input_prompt | query_llm

### クイズ生成するチェーン


class QuestionAndAnswer(BaseModel):
    question: str = Field(description="クイズの問題文")
    answer: str = Field(description="クイズの回答")


quiz_llm = llm.with_structured_output(QuestionAndAnswer)

system = """あなたはクイズの問題考案者です。与えられたドキュメントからそのジャンルに関するクイズを出題します。\n
    クイズのレベルはマニアックですが、回答が単語一語になるような問題にしてください。内容はロジカルにステップバイステップで考えてロジカルに論理構造を理解して作りなさい。"""

input_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "ジャンル:{genre}, 取得されたドキュメント{document}"),
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


### 答えを発表するnode
def tell_answer(state):
    answer = state["answer"]
    question = state["question"]
    message = state["messages"][-1].content
    document = state["document"]

    response = answer_teller.invoke(
        {
            "message": message,
            "answer": answer,
            "question": question,
            "document": document,
        }
    )
    return {"messages": AIMessage(response)}


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

    response = quiz_generator.invoke({"genre": genre, "document": document})
    return {
        "messages": response.question,
        "question": response.question,
        "answer": response.answer,
    }


### ヒントを生成するnode
def hint_generate(state):
    document = state["document"]
    question = state["question"]
    answer = state["answer"]
    messages = state["messages"]

    response = hint_generator.invoke(
        {
            "document": document,
            "question": question,
            "answer": answer,
            "messages": messages,
        }
    )
    return {"messages": AIMessage(response)}


### ユーザーのメッセージを判断するedge
def router(state):

    input_type = input_router.invoke({"messages": state["messages"]}).input_type

    if input_type == "question":
        return "generate question"
    elif input_type == "answer":
        return "tell answer"
    else:
        return "generate hint"


builder = StateGraph(GraphState)

builder.add_node("get_genre", get_genre)
builder.add_node("create_query", create_query)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("quiz_generate", quiz_generate)
builder.add_node("tell_answer", tell_answer)
builder.add_node("hint_generate", hint_generate)

builder.add_conditional_edges(
    START,
    router,
    {
        "generate question": "get_genre",
        "tell answer": "tell_answer",
        "generate hint": "hint_generate",
    },
)
builder.add_edge("get_genre", "create_query")
builder.add_edge("create_query", "search_wikipedia")
builder.add_edge("search_wikipedia", "quiz_generate")
builder.add_edge("quiz_generate", END)
builder.add_edge("hint_generate", END)
builder.add_edge("tell_answer", END)


# このスクリプトファイルの位置を基準に保存先を指定
current_file_path = os.path.abspath(__file__)  # このスクリプトの絶対パス
parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 一つ上の階層
images_directory = os.path.join(parent_directory, "images")  # imagesディレクトリ
output_file_path = os.path.join(images_directory, "mermaid_graph.png")


DB_URI = f"postgresql://{USER_NAME}:{USER_PASS}@127.0.0.1:5432/postgres?sslmode=disable"
print(DB_URI)

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
db_connection = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
checkpointer = PostgresSaver(db_connection)
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)


# サーバーを立ち上げた時にDBとの接続を開始・落とした時に切断
@asynccontextmanager
async def lifespan(app: FastAPI):

    yield

    if db_connection:
        db_connection.close()


app = FastAPI()


class InputRequest(BaseModel):
    user_input: str
    thread_id: str


@app.post("/llm")
async def invoke(request: InputRequest):
    result = graph.invoke(
        {"messages": request.user_input},
        {"configurable": {"thread_id": request.thread_id}},
    )
    return {"responce": result["messages"][-1].content}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
