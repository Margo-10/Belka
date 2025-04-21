FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y gnupg2 lsb-release wget && \
    echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    lsb-release \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libreoffice \
    postgresql-client-17 \
    libfontconfig1 \
    libjpeg-turbo8 \
    libpng16-16 \
    libssl3 \
    libx11-6 \
    libxcb1 \
    libxrender1 \
    xfonts-75dpi \
    xfonts-base \
    git \
    cmake \
    protobuf-compiler \
    libprotobuf-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Установка wkhtmltopdf
RUN wget -q https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.jammy_amd64.deb && \
    dpkg -i wkhtmltox_0.12.6-1.jammy_amd64.deb || apt-get install -f -y && \
    rm -f wkhtmltox_0.12.6-1.jammy_amd64.deb

# Настройка путей CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Копирование файлов проекта
COPY . /app
WORKDIR /app

# Создание рабочих директорий
RUN mkdir -p /app/data/downloaded_files \
    /app/output \
    /app/logs \
    /app/offload

# Настройка окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    POPPLER_PATH=/usr/bin \
    TESSERACT_CMD=/usr/bin/tesseract

CMD ["bash"]