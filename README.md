# Requirements and instructions to run the fe_dr_pm technique

Before delving into the details of the project files, please consider that this project has been executed on a Windows 10 machine with Python 3.8.1. There are a few libraries that have been used within Python modules. Among these, there are:

- tensorflow 2.8.0
- scipy 1.8.0
- scikit-learn 1.0.2
- pm4py 2.2.19.1

Please note that the list above is not comprehensive and there could be other requirements for running the project.

The execution of the technique is triggered by the DOS autorun.bat script. This script sets the parameters required to run the Python programs, clears the environment (which can also be cleaned by executing the clean_environment.bat script separately), copies the normal and anomalous (feature extracted) time series from the Data folder, and chains the execution of dimensionality reduction (dr.py), event log extraction (el_ext.py), behavior characterization (behavior_characterization.py) and anomaly detection (anomaly_detector.py).

autorun.bat requires entering three input parameters by the command line, namely the conformance checking technique to use for anomaly detection, and the number of clusters and window length for event log extraction. The other parameters can be set by editing the batch file manually. Please note that these parameters are set to pre-defined values; you may want to edit them to assess changes in the results.
