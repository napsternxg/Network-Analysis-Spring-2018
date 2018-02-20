# Network Analysis Spring 2018
Network Analysis course

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/napsternxg/Network-Analysis-Spring-2018)

## Installation
Install miniconda with python 3.6 from https://conda.io/miniconda.html
Clone this respository to your machine using either of the following options:

### Using Github (requires installation of git)

* Install git from https://git-scm.com/downloads
* Open your command line tool (CMD for windows and terminal for Mac or Linux)
* Go to a directory where you would like to clone the repository
* Run `git clone https://github.com/napsternxg/NetworkAnalysis_Spring2018.git`

### Using zip files
* Download the latest zipfile of the source code from https://github.com/napsternxg/Network-Analysis-Spring-2018/archive/master.zip
* Unzip the files to a directory


#### WINDOWS USERS ONLY
- Install [GraphViz](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) using the **graphviz-2.38.msi** file.
- Add the location of GraphViz path's bin folder to your user path. E.g. if GraphViz was installed at `C:\Program Files (x86)\GraphViz2.8` then add `C:\Program Files (x86)\GraphViz2.8` to the path. [Help on editing user path](https://www.java.com/en/download/help/path.xml). 

### Install dependencies
* Open your terminal and go to the folder where you have cloned or unzipped the files
* Run the command `conda env create -f environment.yml`. For help refer to [Anaconda managing environment](https://conda.io/docs/using/envs.html)
* The above command should install the required dependencies on your machine including all libraries
* Activate the environment using `source activate na_spring_2018` on Mac/Linux and `activate na_spring_2018` on Windows for help refer to [Activating Environment](https://conda.io/docs/using/envs.html#change-environments-activate-deactivate)
* Once your environment is activated you are ready to run the code.

## Run Notebook environment
From the same directory where you have the cloned or unzipped files, run the following command. 
```
jupyter notebook
```

This should launch a webpage in your browser with the notebook environment.

## TODO before class

* Run the example notebooks and report in the moodle forum if you face any errors.
* The way to run a notebook is to open the notebooks folder in the opened webpage after running `jupyter notebook` command described above.
* Then Click on each notebook. You will see a notebook interface with a menu bar. From the menu bar select `Kernel > Restart and Run All`.
* The above command should run all the cells in the notebook. 
* Do the same for the NetworkX and Mesa notebooks. 
* For the twitter notebook follow the process described below to obtain an API key. You will need to enter this the first time you run this notebook. 
* When you run the Twitter notebook, you will be prompted for some input. The first input will be related to the API keys, please enter that. 
* The next inputs will be general strings to search for. You should enter that as well.
* Wait for the notebook to finish. 
* Report on the forum if you see any kind of error message.

## Examples details
* The notebook [NetworkX.ipynb](./notebooks/NetworkX.ipynb) has some basic code on getting started with NetworkX
* The notebook [Twitter.ipynb](./notebooks/Twitter.ipynb) has the demo code on running twitter based network analysis.
* The notebook [Mesa.ipynb](./notebooks/Twitter.ipynb) has the demo code on running agent based simulation models.

## Getting Twitter API keys
Please refer to [Twitter App Creation](https://dev.twitter.com/oauth/overview/application-owner-access-tokens) page for details on getting your API keys.
