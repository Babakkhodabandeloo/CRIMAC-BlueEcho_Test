import KoronaScript.Modules as ksm
import KoronaScript as ks
import os
import sys
from netCDF4 import Dataset
import glob
import json

"""

Running KoronaScript (CRIMAC-KoronaScript) to produce NetCDFs (broadband averaged over the band and
written as if it was CW)

"""

# =============================================================================
# Surveys to run through
# =============================================================================

# Not necassary, but used here: reads info from a json file
# This file needs to be modified, or just write the parameters in the script
runFiles='/data/crimac-scratch/2025/OF_Conversion/runSalmon.json'

# =============================================================================
# Run Korona
# =============================================================================

# Get input parameters
if os.path.exists(runFiles):
    print(runFiles)
    with open(runFiles, 'r') as f:
        par = json.load(f)
elif len(sys.argv) != 4:
    print(f'Usage: {sys.argv[0]} inputdir outputdir lsss ')
    exit(0)
else:
    par = {"inputdir": sys.argv[1],
            "outputdir": sys.argv[2],
            "lsss": sys.argv[3],
            "ttranges": sys.argv[4]}

# Set lsss env variable
os.environ["LSSS"] = par['lsss']

ksi = ks.KoronaScript(TransducerRanges = par['trranges'])

# Is this necassary?
ksi.add(ksm.ComplexToReal(Active="true")) 

#    ksi.add(ksm.ChannelRemoval(Active = "true",
#                                      Channels = "2",
#                                      KeepSpecified = "true"))

# If needed:
ksi.add(ksm.DataReduction(Active = "true",
                            TransducerRange = "false",
                                MaxRange = "100"))  

# Can be skiped
#    ksi.add(ksm.EmptyPingRemoval(Active = "true"))

# Spesify the "main" frequency which sets the grid
# Can remove angles if you don't want that, WriteAngels = "false"
ksi.add(ksm.NetcdfWriter(Active = "true",
                            MainFrequency="70",
                            WriterType = 'GRIDDED',
                            GriddedOutputType="SV_AND_ANGLES",
                            WriteAngels = "true"))

ksi.write()

ksi.run(src=par["inputdir"], dst=par["outputdir"]) 





