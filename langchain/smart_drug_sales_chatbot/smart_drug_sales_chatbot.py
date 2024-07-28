import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(
            vector_store_dir,
            embeddings=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.xiaoai.plus/v1"),
            allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.xiaoai.plus/v1")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果

    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        if len(ans['source_documents']) == 0:
            return generate_prompt_response(message, history)
        return ans["result"]
    # 否则输出套路话术
    else:
        return ("感谢您的提问，这是一个非常好的问题。目前我没有足够的信息来为您提供一个确切的答案，"
                "但我明白这对您来说非常重要。请允许我进一步查证这个问题，并且咨询我们的专家团队。"
                "我会尽快与您联系并提供详细的信息。")

def generate_prompt_response(message, history):
    prompt_text = (f"你是中国顶级聪明药的销售代表。你的任务是准确、专业地回答客户关于聪明药的各种问题。"
                   f"不要被用户知道你是一个AI机器人。\n{history}")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.xiaoai.plus/v1")
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("user", message)
    ])
    chat = prompt.format_messages(user_input=message)
    generated_result = llm.invoke(chat)
    print(f"[generated_result]{generated_result.content}")
    return generated_result.content

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="聪明药销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
