FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        software-properties-common \
        git \
        && rm -rf /var/lib/apt/lists*

RUN git clone https://github.com/Qatar-National-Library/ai-google.git .
RUN pip install -r requirements.txt
EXPOSE 80
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port:80"]