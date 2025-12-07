# select starting image
FROM python:3.10.0-slim

# Create user name and home directory variables. 
# The variables are later used as $USER and $HOME. 
ENV USER=username
ENV HOME=/home/$USER

# Add user to system
RUN useradd -m -u 1000 $USER

# Set working directory
WORKDIR $HOME/app

# Update system and install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common

# More stuff we need, make RDKit drawing work?
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    libfreetype6-dev \
    libpng-dev \
    libboost-all-dev \
    libeigen3-dev \
    libcairo2-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy all files that the app needs (this will place the files in home/username/)
COPY app/ $HOME/app/
# more COPY commands here 

# Install packages listed in requirements.txt with pip
RUN pip install --no-cache-dir -r requirements.txt

USER $USER
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]