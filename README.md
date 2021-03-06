# sparse-mlp-structure-learning
This is the thesis implementation of Mohamad Wahed Bazo (wahedbazo@outlook.com)
for the degree of B.Sc. Informatik at the Universität Stuttgart.

Thesis title:
**Learning Quantitative Argumentation Frameworks Using Sparse Neural Networks and Swarm Intelligence Algorithms**

##Project Requirements
The project is a python project. The used python version: 3.8.3


To be able to run the project, the following python libraries has to be installed:
* numpy
* pytorch
* sklearn
* pandas
* sparselinear

For the sparselinear library installation, the Pytorch Sparse package has to be preinstalled. Instructions:
https://pypi.org/project/sparselinear/

Note: one needs to pay attentions to the pytorch version before the installation.

##Project structure

The "python" directory contains the followings:
* Three directories for the three datasets that were explored
* "common" directory that contains all common source files, such as the PSO class and the helper script
* all runnable scripts

In each dataset directory, we find four subdirectories:
* dataset: contains all dataset relevant files
* pso: contains the PSO implementation and the "results" directory, which contains output reports and result files.
* mlp: contains the implementation of fully connected MLPs
* decision_tree: contains the implementation and plots of decision trees

The runnable files have the format: 
* <dataset_name>_pso_main.py
* <dataset_name>_sparse_model_evaluation_main.py
* <dataset_name>_mlp_main.py
* <dataset_name>_decision_tree_main.py

In the pso/results directory we find three types of files:
* <experiment_id>.txt: a report file that contains all the applied configurations and hyperparameter values and the results of the experiment
* connections_<experiment_id>_0.txt and connections_<experiment_id>_1.txt: binary connection matrices of the found model by the PSO. They can be used to construct and evaluate the corresponding model
* <experiment_id>.png: a plot of the PSO optimization progress


##Running the scripts from a terminal
To run a specific script, navigate to the project root (.../sparse-mlp-structure-learning/python/) and run the command 'python' and the script name:
```bash
python script_name.py
```

Most results will be printed in the terminal. The PSO results however will be persisted in a report file in the '<dataset_name>_classifiers/pso/results'
directory.
For parameter modification, please navigate to the associated source file stated in the runnable file and modify these directly there.
##Documentation and comments:
Since the implementation is similar among the three dataset, only common sources and the mushroom implementation was documented and commented on in details. Please read these first for information about classes and functions.


##Experiment reproducibility
To reproduce a PSO experiment, read a report file in the pso/results directory and copy the configurations and
parameter values to the <dataset_name>/pso/<dataset_name>_pso_script.py and run the executable file from root director. The experiment identifier string has to be changed so that the old result files are not overridden.
To evaluate a model, modify the paths to the connection files in the <dataset_name>/pso/model_evaluation.py files and run the corresponding executable file.