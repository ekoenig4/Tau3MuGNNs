# Tau3MuGNNs

## 1. Requirements
The project is developed with Python 3.8.5 and it uses the following packages to build models:
```
torch==1.8.1
torch_geometric==1.7.0
```
The full list can be found in `./requirements.txt`.

## 2. Run the code
To train and test a model, please run:
```
cd src
python main.py
```


## 3. File Structure

```
.
├── README.md
├── Tau3MuGNNs.pptx                     Model arch prepresentation
├── data
│   ├── README.md
│   ├── processed                       Store processed .pt data
│   └── raw                             Store raw .pkl data
├── data_visz.ipynb
├── logs
├── requirements.txt
└── src
    ├── configs
    │   └── config7_station1_only.yml   Store configs of the model to be trained
    ├── layers                          Put implemented GNN layers
    │   ├── PlainGAT.py
    │   ├── RelationalGAT.py
    │   └── __init__.py
    ├── main.py                         Train and test the model
    ├── model.py                        Build the model
    └── utils
        ├── __init__.py
        ├── dataset.py                  Create Tau3MuDataset class 
        ├── root2df.py                  Genereate .pkl files from .root files
        └── utils.py                    Store helper functions
```