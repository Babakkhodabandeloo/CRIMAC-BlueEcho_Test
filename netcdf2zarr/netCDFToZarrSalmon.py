# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:24:01 2023

@author: a32685
"""

import subprocess
import dask
import numpy as np
import xarray as xr
import zarr  # as zr
import os.path
import netCDF4
import os
from save_distping import save_distping_parquet
from correct_distping import correct_parquet
from NetCDFInfoExtractor import NetCDFInfoExtractor
import pandas as pd
import pyarrow.parquet as pq
import numcodecs
from numcodecs import Blosc
import dask.array as da
import re

dask.config.set(scheduler='single-threaded')

#dataout='\\data\\crimac-scratch\\tmp\\test_ZoopSeis\\WBAT1\\out\\netcdf'
#OUTPUT_NAME='ZoopSeis_dbl_Rigg1_iw-Phase0'
# dataout='/data/crimac-scratch/tmp/test_ZoopSeis/WBAT2/out/netcdf'
# OUTPUT_NAME='ZoopSeis_dbl_Rigg2_iw-Phase0'
# dataout='/mnt/z/test_data/LoVe/2018/EKProc/out/netcdf'
# OUTPUT_NAME='LoVe_2018_N1.test'
# dataout='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/MonthExample/out'
# OUTPUT_NAME='LoVe_2018_N1.month'

dataout='/data/crimac-scratch/2025/OF_Conversion/temp'
OUTPUT_NAME='_SalmonTest'


ping_time_chunk =10000
range_chunk =2500

outputzarr = dataout+OUTPUT_NAME+'.zarr'
outputfinalzarr= dataout+OUTPUT_NAME+'_sv.zarr'

# Combine nc files
#dataout='Z:\\tmp\\test_ZoopSeis\\WBAT1\\out\\netcdf'
# Define the netcdf folder path and netcdf info output CSV file path
netcdf_folder_path = dataout
netcdf_csv_file = dataout+OUTPUT_NAME+'_netcdf_info.csv'

# Create an instance of the NetCDFInfoExtractor
extractor = NetCDFInfoExtractor(netcdf_folder_path, netcdf_csv_file)

# Extract NetCDF information and write it to a CSV file
file_data = extractor.extract_info()
extractor.write_to_csv(file_data)
    
#save distance and ping_time to a parquet file
outputparquet = dataout+OUTPUT_NAME+'_pingdist.parquet'
save_distping_parquet(dataout, outputparquet)

# #correct distance and ping_time
correct_parquet(outputparquet)

# # Load NetCDF files
# nc_files = [dataout + '\\sv\\' + _F for _F in os.listdir(dataout + '\\sv\\') if _F.endswith('.nc')]
nc_files = [dataout + '/sv/' + _F for _F in os.listdir(dataout + '/sv/') if _F.endswith('.nc')]
nc_files.sort()
print('Reading nc files')

corrected_distping_path = outputparquet.replace(".parquet", "corrected.parquet")

# # Read the Parquet file with corrected distance and ping_time into a DataFrame
updated_distping = pd.read_parquet(corrected_distping_path)

# # Convert the DataFrame columns to Dask arrays with the same chunk sizes as the rest of the data
ping_time_array = da.from_array(updated_distping['ping_time'].values, chunks=ping_time_chunk)
distance_array = da.from_array(updated_distping['distance'].values, chunks=ping_time_chunk)

# # Create xarray DataArrays from Dask arrays
ping_time_xarray = xr.DataArray(ping_time_array, dims=['ping_time'])
distance_xarray = xr.DataArray(distance_array, dims=['ping_time'])

chunk_sizes_obj = {
    'channel_id': {'frequency': 1},
    # Add more variables and their chunk sizes as needed
}

# Use open_mfdataset to concatenate the files lazily with chunking
ds = xr.open_mfdataset(
   nc_files,
   combine='by_coords',
   decode_times=True,  # Assuming 'ping_time' is a time-like variable
).chunk({'ping_time': ping_time_chunk, 'range': range_chunk})


# This approach did not work due to conflicts in the 'range' variable across files
# # Use open_mfdataset to concatenate the netCDF files, providing the corrected ping_time
# ds = xr.open_mfdataset(
#     nc_files,
#     combine='by_coords',
#     decode_times=True,  # Assuming 'ping_time' is a time-like variable
#     coords={'ping_time': ping_time_xarray},  # Provide corrected ping_time as a coordinate
#     chunks=chunk_sizes_obj,  # Specify chunk sizes for problematic variables
#     compat='override',  # Skip check for conflicting variables
#     ).chunk({'ping_time': ping_time_chunk, 'range': range_chunk})
# print(ds)

# Alternative approach:
# 1. Concatenate files by stacking them along the 'ping_time' dimension.
#    This is more robust to inconsistencies in other coordinates like 'range'?
# ds = xr.open_mfdataset(
#     nc_files,
#     combine='nested',
#     concat_dim='ping_time',
#     decode_times=True,
#     compat='override',  # Good to keep for other minor conflicts
#     chunks={'ping_time': ping_time_chunk, 'range': range_chunk}
# )

# ds = xr.open_mfdataset(
#     nc_files,
#     combine='nested',
#     concat_dim='ping_time',
#     decode_times=True,
#     compat='override'
# )

ds = ds.chunk({"frequency": 1, "range": range_chunk, "ping_time": ping_time_chunk})

print(ds)

# Quick check to ensure dimensions match before assignment
if len(ping_time_xarray) != len(ds['ping_time']):
    raise ValueError(
        f"Corrected ping_time length ({len(ping_time_xarray)}) does not match "
        f"concatenated data length ({len(ds['ping_time'])}). "
        "Please check the logic in your parquet file creation."
    )

version='1.0.1'

# Update the corresponding variables in the dataset
ds['ping_time'] = ping_time_xarray
ds['distance'] = distance_xarray

# Convert specific Dask arrays to fixed-size dtype
ds['ping_time'] = ds['ping_time'].astype('datetime64[ns]')  # Assuming datetime dtype
ds['distance'] = ds['distance'].astype(np.float64)

# Change data types of channel_id and raw_file
ds['channel_id'] = ds['channel_id'].astype('<U38').chunk({'frequency': 1})
ds['raw_file'] = ds['raw_file'].astype('<U29')

ds['pulse_length'] = ds['pulse_length'].transpose('frequency', 'ping_time')

git_rev = os.getenv('COMMIT_SHA', 'XXXXXXXX')
print(git_rev)
# Update attributes
ds.attrs['git_commit'] = git_rev
ds.attrs['name'] = 'Dockerized KORONA'
ds.attrs['KORONA version'] = ds.attrs.pop('version')
ds.attrs['scriptversion'] = version

print(ds)


print("-------- test -------")
print("-- updated distance--")
print(ds['distance'] )
subset_of_distance_values = ds['distance'][:10].compute()
print("---")
print(subset_of_distance_values)
print("---------------------")

ds = ds.chunk({"frequency": 1, "range": range_chunk, "ping_time": ping_time_chunk})
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
encoding = {var: {"compressor": compressor} for var in ds.data_vars}

ds.to_zarr(outputzarr, mode="w", encoding=encoding)
zarr.consolidate_metadata(outputzarr)
#print("move " + outputzarr+ " " + outputfinalzarr )
os.system("move " + outputzarr+ " " + outputfinalzarr )

print("finished writing zarr")

# # Delete nc temporary files only for STEP=0
# if os.getenv('STEP') == '0':
#     er = [os.remove(_nc_files) for _nc_files in nc_files]
#     os.rmdir(dataout+'sv/')
