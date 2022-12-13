## Directory Structure:

The file `paths.json` contains the following paths that you should edit:

* "models_path": The directory where you want to store your saved models
* "log_path": Directory to redirect output
* "data_dir": Data loading for training / testing
* "graph_dir": Where to save any graphs / images generated from the models
* "prediction_dir": Predicted output for each model

The files borrowed from Rocheteau Et al. are included in the folder 'eICU_preprocessing'.
All files outside of  this folder were produced by Elishua K Shumpert and Stephen Toner for
 SI 670: Applied Machine Learning at the University of Michigan, Fall 2022.

Because the dataset in question requires credentialed access, we have deleted the data from
eICU to comply with our right to use the data. To reproduce results, place all files from the eICU 
database (or demo-database) in the 'data' folder.
