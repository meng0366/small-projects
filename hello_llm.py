"""环境自检脚本：跑通第一个 LLM 调用（对应步骤 S0.6）。

运行前请先配置 .env 中的 DEEPSEEK_API_KEY。
运行：python hello_llm.py
"""
from app.config import check_config, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def main():
    if not check_config():
        print("[未就绪] 请先把 .env.example 复制为 .env，并填入真实 API Key。")
        return

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )
    resp = llm.invoke("用一句话介绍你自己")
    print("模型回复：", resp.content)


if __name__ == "__main__":
    main()
