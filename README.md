# Tau3MuGNNs
Application of Graph Neural Networks to the study of the tau->3mu signature at the HL-LHC CMS L1 Trigger

## Install
```
git clone https://github.com/cms-p2l1trigger-tau3mu/Tau3MuGNNs 
````

## Instructions to run the first script
The first script reads the simulation root files and covert them into NumPy array and save them as a pickle file (*.pkl). This script was tested in Python3 using the following dependencies:

-uproot3, 3.14.4

-numpy, 1.15.4

-pandas, 0.24.2

To install these dependencies (if needed)

```
pip3 install --user uproot3 pandas numpy
````

First, we should locate the ROOT files (DsTau3muPU0_Private.root,DsTau3muPU200_MTD.root,MinBiasPU200_MTD.root) in the input folder myrootfiles. Then, one run the script ProcessROOTFiles.py using the below instruction, where processing.cfg defines the configuration of the script (e.g. samples, variables, etc). The outputfiles will be stored as *.pkl files in the outputfolder myoutputfiles.

```
python3 ProcessROOTFiles.py --config config/processing.cfg
````
One can specify the maximum number of entries to proccess in the ROOT files using the option --maxevts (by default 100000) 