# Use an official Python runtime as the base image
FROM python:3.10.16-slim

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --quiet -r requirements.txt

# Install tabpfn-extensions from the git repository
#RUN pip install --quiet "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"

# Copy the environment file into the container
COPY .env /

# Keep the container running indefinitely
#CMD ["tail", "-f", "/dev/null"]