"""Streamlit Web UI 入口（对应功能 F4.1，阶段 4 实现）。

阶段 1、2 先用命令行；阶段 4 改造成 Streamlit 界面。
运行（阶段 4 起）：streamlit run app/main.py
"""
import streamlit as st


def main():
    st.set_page_config(page_title="数据分析 Agent", page_icon="📊")
    st.title("数据分析 Agent")
    st.info("界面开发中，将在阶段 4 实现。当前请使用 hello_llm.py 检查环境。")


if __name__ == "__main__":
    main()
