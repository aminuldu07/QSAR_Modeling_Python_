# "__Developing Mechanism-Based Models for Complex Toxicology Study Endpoints Using Standardized Electronic Submission Data__"


In this repository is a series of notebooks for modeling complex toxicity endpoints created from Standard for Exchange of Non-clinical Data (SEND) datasets.  In these notebooks, using hepatoxicity as a motivating endpoint, methods are being developed to develop novel data-mining techniques to identify correlations between biomarkers (clinical chemistry tests, animals sex or body weights) and various signs of drug-induced toxicity.  In this series of notebooks, three drug-induced liver injury (a.k.a hepatoxicity) phenotypes are used as a proof of concept, steatosis, cholestasis, and hepatocellular necrosis to data mind a SEND db.  This method could easily be extended to other toxicity endpoints, as needed.  

The basic steps of the methodology thus far are:

1) Identify a target species of animals (e.g., RAT or DOG).  
2) Transform clinical chemistry tests and animal meta data to be used as "features".  
3) Classify histopathology findings as disease phenotypes (steatosis, cholestasis, necrosis) to be used as a training classes/labels.  
4) Build and validate machine learning models trying to use features to predict classes (i.e, animal clin chem tests to predict disease).  
5) Evaluate features relevate to different disease phentoypes to speculate on biological mechanism.   


## Getting started

### Configuration - creation of SQLite database

Before running these notebooks a SQLite database of SEND datasets needs to be created.  Currently, the FDA stores datasets on the Electronic Document Resources Server, also known as the EDR.  In order to load these into a SQLite lite database they must be (1) first identified and aggregated to the user's local laptop, then (2) loaded into a SQLite database.  The first can be accomplished by a simple powershell script located in this repository named `copy_send_ds.ps1` located in the `scripts/` folder.  The second can be done via an R script in the [biocelerate git repository](https://github.com/phuse-org/BioCelerate) called `poolAllStudyDomainsStrict.R`. 

This process has been done previously and, as of May 2020, a database containing 1,895 SEND datasets is located at `/home/daniel.russo/data/send/send_3_4.db`. 


### Configuration - variable set up

Regardless whether this or a newer version of the SQLite send database is created, the database with which these notebooks will use needs to be configured.  This can be done in the python script `send.py` at the very top.  This script requires two variables to be configured, `db_dir` and `send_db_file`.  `db_dir` should point to the directory that contains the SQLlite database and `send_db_file` should be the SQLite database file itself. 


### Configuration - conda environment

A dedicated conda environment called `cheminformatics` is set up on the HIVE server which has all the external libraries needed for these notebooks.  Additionally, there is an `environmental.yml` file which can be used to create your own conda environment for these notebooks by running the command:

```
conda env create -f environmental.yml
```

## Overview - table of contents

### Workflow scripts

These notebooks layout the basic workflow of the entire project, and thus, should mostly be run in order.  

After configuration the notebooks could be run in order each with it's own purpose:

1) Creation of a training set from the SEND database, given a species in controlled terminology (e.g., RAT or DOG).  
2) The creation of partial least squares models coupled to a logistic function.   
3) Merging and analyzing the trained model prediction results.   
4) Calculating "z score" profiles for each hepatotoxicity phenotypes.   
5) Calculating percent feature residuals for each hepatotoxicity phenotypes.     

### Plotting Scripts

These notebooks are just for creating figures and such.  Generally, they're not specific to this project and can be used to create things like SEND frequency counts, making eDISH plots from the SEND data, things liket that....  



### TODO

1) Figure out `send.py` and relative imports.  