FROM python:3.12-slim

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements file first to leverage Docker cache
COPY --chown=user:user requirements.txt $HOME/app/

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . $HOME/app/

# Hugging Face Spaces port
ENV PORT=7860
EXPOSE 7860

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
