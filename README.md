
# DirectSearchApplication

This repository contains the code for running Application for Direct Search. Follow the instructions below to set up and run the simulations.

## Directory Structure

```sh
DirectSearchSim/
├── config.yaml
├── data/
├── environment.yml
├── models/
├── README.md
├── DSApp.py
└── utils.py
```

## Setup Instructions

### Prerequisites

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.

### Clone the Repository

Clone the repository using the following command:

```sh
git clone https://github.com/nilson01/SimulationDirectSearch.git
cd ApplicationDirectSearch
```

### Install Dependencies

1. **Create the Conda environment:**

   ```sh
   conda env create -f environment.yml
   ```

2. **Activate the environment:**

   ```sh
   conda activate DirectSearchApplication
   ```

## Configuration

Before running the simulations, you need to edit the `config.yaml` file according to the setup you want to check.

## Running the Application

The main Application code is in the `DSApp.py` file. This file also includes parallelization for efficient execution.

### Run with Parallelization

To run the simulation with parallelization, use:

```sh
python DSApp.py
```

### Run a Simple Single Simulation

If you don't want parallelization or want to run a simple single simulation, you can edit the `DSApp.py` file to disable or modify the parallelization settings.

### Utility Functions

Utility functions required for the simulations are available in the `utils.py` file.


## Example Usage

1. **Edit `config.yaml`**:
   Modify the `config.yaml` file to set up the parameters for your simulation.

2. **Run the simulation**:
   ```sh
   python DSApp.py
   ```

3. **Check the logs**:
   Logs will provide information in detail of the simulation.

## Contact

For any issues or questions, please contact [Nilson Chapagain](mailto:nilson.chapagain@gmail.com).
```

# ApplicationDirectSearch
