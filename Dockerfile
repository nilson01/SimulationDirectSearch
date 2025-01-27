FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the Conda environment YAML file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment in the container shell
SHELL ["conda", "run", "-n", "DirectSearchApplication", "/bin/bash", "-c"]

# Copy your project files
COPY . .

# Set the default command to run your script
CMD ["conda", "run", "-n", "DirectSearchApplication", "python", "DSAppSim.py"]
