# AeroCirTree
A system for airport weather forecasting based on circular regression trees

### About:
**AeroCirTree** is a Python 3.x package and a set of command line tools to extract historical weather timeseries for airports, train regression tree models and test the results.
The Python package is built around the `Data` and `Node` classes and the functionality that it provides can be incorporated in other programs. There are three command line interface scripts, named **aerocirtree_extract**, **aerocirtree_train** and **aerocirtree_test** which make use of this library to fetch historical timeseries, train models and test results for any airport in the world.

### Requirements:

This program is written in Python and to run it requires version 3. It also requires the Numpy and Pandas packages to be available in your environment.

If you don't already have a Python environment set up which fulfills these requirements, the easiest way to start is by using the Miniconda package manager.

http://conda.io

Once you have set up a Python 3 environment enter the following commands to install the required dependencies:

```bash
conda install pandas
```

```bash
conda install numpy
```

Once your environment is set up, clone this repository and access its directory.

### Howto:

Extracting historical data for an airport:

The script **aerocirtree_extract** is used to extract the historical time series data, in csv format, for any airport in the world. This command has three flags to state the name of the airport (ICAO code), starting and end dates in YYYYMMDD format. For example to extract data for the airport of London Heathrow for the first six months of 2016:

```bash
./aerocirtree_extract --airport EGLL --start_date 20160101 --end_date 20160601
```

To save the output in a file, Unix pipes can be used to redirect the output into a file (supposing a Unix like environemnt):

```bash
./aerocirtree_extract --airport EGLL --start_date 20160101 --end_date 20160601 > EGLL.csv
```

Training a regression tree model:

The script **aerocirtree_train** is used to train a model based on a specified data set. The configuration options for the model are specified using an external JSON file. An example of this file looks like

```bash
{"output":"metar_temp",
    "input":[{"name":"gfs_wind_dir","type":"circular"}, 
             {"name":"gfs_rh","type":"linear"}],
    "contiguous":true
    "max_leaf_size":100}
```

This file specifies the name of the output variable, the list of variables used as input and the nature of these variables being either "linear" or "circular". This document also contains the tree splitting methodology being "contiguous" or "non-contiguous" as well as the maximim leaf size value used as the stop criterium for the tree.

If the data set is in a file called **EGLL.csv** and the config file is in a file called **Model_A.json** the command to train the model is:

```bash
./aerocirtree_extract --data EGLL.csv --config Model_A.json
```

Running program would save a model on the local directory called **EGLL_Model_A.mod**

The program is distributed with some data sets which has been already downloaded inside the **datasets** directory. This files allow users to experiment with training models without the need of having to download their own data, which for long time series can be a slow process. The test folder contains some examples of config files for different trees. For example, to train a model using the data extracted for Sydney airport and the config file specifying a model to forecast 10-meter wind speed we would do:

```bash
./aerocirtree_train --data datasets/yssy.csv --config test/wspd_2cir_press.json
```

### Manuscript experiment:

To reproduce the results presented in the original manuscript, use this script that runs the different versions of the regression trees and outputs the results formated as a markdown table:

`python manuscript_results.py test/wspd_2cir_press.json`

