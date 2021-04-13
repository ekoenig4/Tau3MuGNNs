import numpy as np
import pandas as pd
import uproot3 as uproot
import argparse
import configparser
import ast
import pickle

def SaveFileAsPickle(file,outputfile):
    with open('%s'%outputfile, 'wb') as output:
      pickle.dump(file, output)

def LoadPickleFile(picklefile):
    with open('%s'%picklefile, "rb") as fp:
      file = pickle.load(fp)
    return file 

def ProcessingROOTFile(samplename,variables,inputdirectory,outputdirectory,is_signal,maxevts):
    ## open dataset
    print("\n... Opening file in input directory using uproot: %s.root"%samplename)
    inputname  = '%s/%s.root'%(inputdirectory,samplename)
    events     = uproot.open(inputname)['Ntuplizer/MuonTrackTree']
    
    #transform file into a pandas dataframe
    print("\n... Processing file using pandas")
    unfiltered_events_df   = events.pandas.df(variables, entrystop=maxevts,flatten=False)

    # Save numpy array (or dataframe as well!) as a pickle
    print("\n... Saving file in output directory: %s.pkl"%samplename)
    SaveFileAsPickle(unfiltered_events_df,'%s/%s.pkl'%(outputdirectory,samplename))
   
    # Load pickle file (if needed or to cross check)
    unfiltered_events_df_loaded = LoadPickleFile('%s/%s.pkl'%(outputdirectory,samplename))
    print(unfiltered_events_df_loaded)


#############COMMAND CODE IS BELOW ######################

###########OPTIONS
parser = argparse.ArgumentParser(description='Command line parser of skim options')
parser.add_argument('--config',    dest='cfgfile',   help='Name of config file',   required = True)
parser.add_argument('--maxevts',   dest='maxevts',   help='Maximum number of events processed',default=100000)

args = parser.parse_args()
configfilename = args.cfgfile
maxevts        = args.maxevts

###########Read Config file
print("\n[INFO] Reading skim configuration file: ",configfilename)
cfgparser = configparser.ConfigParser()
cfgparser.read('%s'%configfilename)
inputdirectory           = ast.literal_eval(cfgparser.get("general","inputdirectory"))
outputdirectory          = ast.literal_eval(cfgparser.get("general","outputdirectory"))
signalsamples            = ast.literal_eval(cfgparser.get("general","signalsamples"))
backgroundsamples        = ast.literal_eval(cfgparser.get("general","backgroundsamples"))
signalvariables          = ast.literal_eval(cfgparser.get("filter", "signalvariables"))
backgroundvariables      = ast.literal_eval(cfgparser.get("filter", "backgroundvariables"))

##### Print the configuration
print("\n[INFO] Information contained in configuration file")
print(" - Input directory:", inputdirectory)
print(" - Output directory:", outputdirectory)
print(" - Input files:")
for sample in signalsamples: print("  **",sample)
for sample in backgroundsamples: print("  **",sample)
print(" - Variables added in the output file:")
for variable in signalvariables: print("  **",variable)

###Process the input ROOT files
print("\n[INFO] Transforming ROOT files into pickle files")
for sample in signalsamples:  ProcessingROOTFile(sample,signalvariables,inputdirectory,outputdirectory,True,maxevts)
for sample in backgroundsamples:  ProcessingROOTFile(sample,backgroundvariables,inputdirectory,outputdirectory,False,maxevts)