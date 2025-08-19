
#
# Assuming a dataset has been prepared and available (xarray/zarr)
# i.e. the CRIMAC preprocessing pipeline has been run (raw->zarr)
# Here we use a test data set (two files) from the annual 2024 NVG herring spawning survey
# The data includes clear herring schools midwater on the shelf outside Vesterålen
#

# Typical steps
# Bottom detection (not performed here yet)
# Remove samples bellow the seafloor (not performed here yet)
# Potential noise reduction (remove spikes etc.) (not performed here yet)
# Resample/average/smooth the two variables
# Calculate dB difference
# Select data that meets the specified "dB difference" criteria (not performed here yet)

#
# Some other important factors include common observation range (ie higher frequencies have
# shorter observation range due to attenuation)
#

import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

local = 0 # Running tests on local machine, 0->on IMR server
average_data = 1 # 1->Average data, 0->No averaging
freq1 = 70000.   # Frequency 1
freq2 = 70000.  # Frequency 2 to substract from frequency 1

# Frequency selection (70 kHz)
freq = 70000.

# Time selection
start_time = "2018-03-04 00:01:25" # Select subset of data
end_time   = "2018-03-04 02:01:30" # Select subset of data


# Example dataset with herring schools, two (crimac-scratch/test_data/dBDiff)
# No preprocessing in Korona (ie averaging)
# Assumes regular grid across frequencies
if local==1:
    f = '/mnt/z/test_data/dBDiff/ACOUSTIC/GRIDDED/out/S2024204001_sv.zarr'
elif local==0:
    # f = '/data/crimac-scratch/test_data/dBDiff/ACOUSTIC/GRIDDED/out/S2024204001_sv.zarr'
    f='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/DayExample/out/netcdfLoVe_2018_N1.test.zarr'

freqs=[freq1, freq2]
print(f)
data=xr.open_dataset(f,engine="zarr")

print('type(data)    : ',type(data) )
print(data.coords)
print(data.data_vars)

ping_times = data['ping_time']
print(ping_times)
print(data['frequency'])


# Select sv between times at 70 kHz
sv_sel = data['sv'].sel(
    ping_time=slice(start_time, end_time),
    frequency=freq
)
sv_sel_db = 10 * np.log10(sv_sel)

dataSel=sv_sel.resample(ping_time="1min").mean(dim=["ping_time"]).coarsen(range=10, boundary="trim").mean()

# Set min/max for visualization
vmin = -82
vmax = -50

fig, ax = plt.subplots(figsize=(12, 6))

# xarray plotting
sv_sel_db.plot.pcolormesh(
    x='ping_time', 
    y='range', 
    ax=ax, 
    cmap='viridis', 
    vmin = vmin,
    vmax = vmax

)

ax.set_title('Sv at 70 kHz (dB)')
ax.set_xlabel('Time')
ax.set_ylabel('Range (m)')
# ax.set_ylim(0, 250)       # optional: limit range
ax.invert_yaxis()         # optional: depth increasing downwards

# plt.show()
fig.savefig('sv_70kHz.png', dpi=150)
plt.close(fig)




# # Subset for ping_time
# data_time = data.sv.sel(ping_time=slice(start_time, end_time))

# # Subset for the two frequencies
# dataSel = data_time.sel(frequency=freqs[0])

# print(' type(dataSel)   : ',type(dataSel))

# # This is a slow operation, can it be improved?
# # What's the best approach? - Coarsen in both dimentions (ping no and range)?
# if average_data==1:
#     dataSel=dataSel.resample(ping_time="1min").mean(dim=["ping_time"]).coarsen(range=10, boundary="trim").mean()
#     #dataSel=dataSel.resample(ping_time="1min").mean(dim=["ping_time"])
#     #dataSel=dataSel.resample(range=10).mean(dim=["range"])
#     # Need a recent xarray version

# # Calculate the difference between the frequencies 38 and 200
# dB_diff = 10*np.log10(dataSel.sel(frequency=freqs[1])) #- 10*np.log10(dataSel.sel(frequency=freqs[0]))

# print('HERE')
# # Select 38 kHz data and plot
# dB_0=10*np.log10(dataSel.sel(frequency=freqs[0]))
# dB_1=10*np.log10(dataSel.sel(frequency=freqs[1]))

# # Visualization of the data and results on differencing

# # Set min and max for visualization of Sv
# vmin=-82
# vmax=-30

# fig, ax = plt.subplots(figsize=(10, 6))
# dB_0.plot.pcolormesh(x='ping_time', y='range', ax=ax, cmap='viridis', vmin=vmin, vmax=vmax)
# ax.set_title('Sv at ' + str(freqs[0]) + ' Hz')
# ax.set_xlabel('Time')
# ax.set_ylabel('Range (m)')
# ax.set_ylim(0,250)
# ax.invert_yaxis()
# # plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))
# dB_1.plot.pcolormesh(x='ping_time', y='range', ax=ax, cmap='viridis', vmin=vmin, vmax=vmax)
# ax.set_title('Sv at ' + str(freqs[1]) + ' Hz')
# ax.set_xlabel('Time')
# ax.set_ylabel('Range (m)')
# ax.set_ylim(0,250)
# ax.invert_yaxis()
# # plt.show()
# plt.close()

# # Plot dB diff
# fig, ax = plt.subplots(figsize=(10, 6))
# dB_diff.plot.pcolormesh(x='ping_time', y='range', ax=ax, cmap='seismic')
# ax.set_title('dB difference ' + str(int(freqs[0]/1000)) + ' kHz - ' + str(int(freqs[1]/1000)))
# ax.set_xlabel('Time')
# ax.set_ylabel('Range (m)')
# ax.set_ylim(0,250)
# ax.invert_yaxis()
# # plt.show()
# plt.close()

# # # Plot with min/max dBdiff values
# # vmin=0
# # vmax=10

# # fig, ax = plt.subplots(figsize=(10, 6))
# # dB_diff.plot.pcolormesh(x='ping_time', y='range', ax=ax, cmap='viridis',vmin=vmin, vmax=vmax)
# # ax.set_title('dB difference / 200-38')
# # ax.set_xlabel('Time')
# # ax.set_ylabel('Range (m)')
# # ax.set_ylim(0,250)
# # ax.invert_yaxis()
# # plt.show()

# # # Cut bellow a set threshold
# # vmax=10

# # fig, ax = plt.subplots(figsize=(10, 6))
# # dB_diff.where(dB_diff<=vmax).plot.pcolormesh(x='ping_time', y='range', ax=ax, cmap='viridis')
# # ax.set_title('dB difference / 38-200')
# # ax.set_xlabel('Time')
# # ax.set_ylabel('Range (m)')
# # ax.set_ylim(0,250)
# # ax.invert_yaxis()
# # plt.show()  
