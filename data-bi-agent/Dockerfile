# 数据分析 Agent —— 单镜像部署（Streamlit）
#
# 构建：docker build -t data-bi-agent .
# 运行：docker run --rm -p 8501:8501 -e DEEPSEEK_API_KEY=sk-xxx data-bi-agent
# 打开：http://localhost:8501
#
# 用 requirements.lock.txt 而不是 requirements.txt：锁定全部传递依赖，保证构建可复现。
FROM python:3.12-slim

# 中文字体：Plotly 出图与页面渲染中文需要，缺字体会显示方块
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖再拷代码，改代码时不必重装依赖（利用构建缓存）
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY . .

# 容器内以非 root 运行
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

EXPOSE 8501

# 健康检查用 Python 而不是 curl：slim 镜像里没有 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
