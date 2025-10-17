# Base Image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set initial working directory
WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/simonucl/spiral

# Set the working directory to the cloned repo
WORKDIR /workspace/spiral

RUN apt-get update && apt-get install -y python3-pip vim && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Ensure Python is in PATH
ENV PATH="/usr/bin:${PATH}"

# Upgrade pip and install core Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
# NOTE: These were separate layers in the history but are combined here for better practice.
RUN pip install --no-cache-dir vllm==0.10.0 && pip install --no-cache-dir oat-llm

RUN pip install -e .

RUN pip install -U vllm==0.10.0

# Set the default command to start a bash shell
CMD ["/bin/bash"]