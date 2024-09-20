## DerivaPredict

DerivaPredict provides an accessible tool to explore potential derivatives with pharmacological activity. 
It streamlines the process by integrating advanced knowledge-driven and deep-learning algorithms for 
derivative structure prediction, compound protein affinity estimation, and ADMET property analysis, 
allowing users to efficiently assess the drug-like potential of novel compounds.


### Installation

To get started with this project, you'll need to create a virtual environment and install the necessary software packages. 
Follow these simple steps, even if you're new to coding:    

1. **Install Conda**: If you don’t have Conda installed, you can download it from [here](https://www.anaconda.com/download/). 
Conda is a tool that helps manage different versions of Python and the packages needed for different projects.    

2. **Create a Virtual Environment**: Open your terminal (or command prompt) and run the following 
command to create a virtual environment named `DerivaPredict`:    

```
conda create -n DerivaPredict python=3.10
```

This creates an isolated space where the project’s required tools and packages will be installed, 
without interfering with your system's Python setup.

3. **Activate the Environment**: Once the environment is created, activate it using the following command:

```
conda activate DerivaPredict
```

4. **Clone the Project Repository**: Download the project’s code from GitHub by running the following command:

```
git clone https://github.com/hcji/DerivaPredict.git
cd DerivaPredict
```

5. **Unzip Dependency**: If you don't already have 7-Zip installed, download and install it from [here](https://www.7-zip.org/).

To make it easier to run 7z commands from anywhere, you can add 7-Zip to your system's PATH:

- Open System Properties > Advanced > Environment Variables.    
- Under System Variables, find Path, click Edit, and add the path where 7-Zip is installed (usually C:\Program Files\7-Zip).

Use the following command to unzip the file:
```
7z e biotransformer.7z
```

6. **Install Dependencies**: The project relies on certain software libraries, 
which are listed in a file called requirements.txt. To install them, navigate into the project folder and run:

```
pip install -r requirements.txt
```

### Usage

1. Open the GUI:

```
python NPDS.py
```

2. Refer the following video:



### Contact
jihongchao@caas.cn




