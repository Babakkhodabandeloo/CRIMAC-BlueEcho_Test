
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
import time
import holoviews as hv
import hvplot.xarray  # enables hvplot on xarray
hv.extension('matplotlib')  # use Matplotlib backend

start = time.time()

local = 0 # Running tests on local machine, 0->on IMR server
average_data = 1 # 1->Average data, 0->No averaging
freq1 = 70000.   # Frequency 1
freq2 = 70000.  # Frequency 2 to substract from frequency 1

# Frequency selection (70 kHz)
freq = 70000.
Threshold = -66 # (dB) Filter the data and ignore data below Threshold (dB)

# Time selection
start_time = "2018-03-04 00:01:25" # Select subset of data
end_time   = "2018-03-05 00:00:25" # Select subset of data

# Range selection
start_range = 50 # m
end_range   = 220 # m

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
# data=xr.open_dataset(f,engine="zarr")
# data = xr.open_dataset(f, engine="zarr", chunks={"ping_time": 100, "range": 500})

# print('type(data)    : ',type(data) )
# print(data.coords)
# print(data.data_vars)

# ping_times = data['ping_time']
# print(ping_times)
# print(data['frequency'])

# 


# Select sv between times at 70 kHz
# sv_sel = data['sv'].sel(
#     ping_time=slice(start_time, end_time),
#     range=slice(start_range,end_range),
#     frequency=freq
# )
sv_sel = (
    xr.open_dataset(
        f,
        engine="zarr",
        chunks={"ping_time": 1000, "range": 5000}
    )['sv']
    .sel(
        ping_time=slice(start_time, end_time),
        range=slice(start_range, end_range),
        frequency=freq
    )
)

Delta_R = float(sv_sel['range'].diff('range').isel(range=0))
print('DeltaR = ', Delta_R)

# sv_sel_db = 10 * np.log10(sv_sel)
sv_sel_db = 10 * xr.apply_ufunc(
    np.log10,
    sv_sel,
    dask='allowed'  # <-- this enables lazy computation with dask
)

# create mask from sv_sel_db
Mask = sv_sel_db > Threshold

# apply mask to original sv_sel, force zeros where mask is False
sv_sel_threhholds = sv_sel.where(Mask, 1E-30)

sv_sel_db = sv_sel_db.where(Mask, -300)
sv_sel = sv_sel_threhholds


# sv_sel_loaded = sv_sel.load()  # converts to in-memory numpy array
# sv_sel_db = 10 * np.log10(sv_sel_loaded)
# print('type(sv_sel),  type(sv_sel_threhholds)', type(sv_sel), type(sv_sel_threhholds))
# print(sv_sel)

# dataSel=sv_sel.resample(ping_time="1min").mean(dim=["ping_time"]).coarsen(range=10, boundary="trim").mean()


# Set min/max for visualization
vmin = -82
vmax = -60

# fig, ax = plt.subplots(figsize=(12, 6))

# # xarray plotting
# sv_sel_db.plot.pcolormesh(
#     x='ping_time', 
#     y='range', 
#     ax=ax, 
#     cmap='viridis', 
#     vmin = vmin,
#     vmax = vmax
# )

# ax.set_title('Sv at 70 kHz (dB)')
# ax.set_xlabel('Time')
# ax.set_ylabel('Range (m)')
# # ax.set_ylim(0, 250)       # optional: limit range
# ax.invert_yaxis()         # optional: depth increasing downwards

# # plt.show()
# fig.savefig('sv_70kHz.png', dpi=150)
# plt.close(fig)

# Create plot
plot = sv_sel_db.hvplot(
    x='ping_time',
    y='range',
    cmap='viridis',
    clim=(vmin, vmax),
    invert_yaxis=True,
    width=2000,
    height=800,
    xlabel='Time',
    ylabel='Range (m)',
    title='Sv at 70 kHz (dB)',
    clabel='Sv (dB)',
    fontsize={'title': 24, 'labels': 18, 'xticks': 16, 'yticks': 16,'cticks':14, 'clabel':14}  # increase font sizes
)


# Save as PNG, PDF, or SVG
hv.save(plot, 'sv_70kHz.png')  # PNG
# hv.save(plot, 'sv_70kHz.pdf')  # PDF
# hv.save(plot, 'sv_70kHz.svg')  # SVG


end = time.time()
print(f"Runtime before Urmy parameter Calcs: {end - start:.2f} seconds")

# Urmy parameters: =============================================
# sv_values = sv_sel.values
# print(sv_values.shape)  # (40, 76)
# print(len(sv_values[0]))

# sum across the range dimension (i.e. collapse depth bins into one value per ping)
sv_sum_range = sv_sel.sum(dim="range")

# From Urmy parameters Urmy et al 2012 - ICES J Marine Science
Integrate_sv_dz = ( Delta_R * sv_sum_range ) # as a function of ping time
# Abundance = 10*np.log10( Integrate_sv_dz ) # as a function of ping time

# Abundance in dB using xarray's built-in log10
# Abundance
Abundance = xr.apply_ufunc(
    np.log10, 
    Integrate_sv_dz,
    dask='allowed'
) * 10

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, Abundance)
ax.set_title('Abundance')
ax.set_xlabel('Ping Time (d HH:MM)')
fig.savefig('Abundance.png', dpi=150)
plt.close(fig)


# Density = 10*np.log10( Integrate_sv_dz/(end_range - start_range) )
# Density
Density = xr.apply_ufunc(
    np.log10, 
    Integrate_sv_dz / (end_range - start_range),
    dask='allowed'
) * 10


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, Density)
ax.set_title('Density')
ax.set_xlabel('Ping Time (d HH:MM)')
fig.savefig('Density.png', dpi=150)
plt.close(fig)

# Range_val = sv_sel.coords['range'].values
# print(sv_sel["range"].shape)
# print(sv_sel.values.shape)

z_product_svz = sv_sel * sv_sel["range"]
z_product_svz_dz = z_product_svz * Delta_R

# print(z_product_svz)

CenterofMass = (z_product_svz_dz.sum(dim="range") ) / Integrate_sv_dz

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, CenterofMass)
ax.set_title('CenterofMass')
ax.set_xlabel('Ping Time (d HH:MM)')
fig.savefig('CenterofMass.png', dpi=150)
plt.close(fig)


# Z_minus_CM_mult_svzdz = ( sv_sel * ((sv_sel["range"] - CenterofMass )**2) ) * Delta_R
# Inertia = (Z_minus_CM_mult_svzdz.sum(dim="range") ) / Integrate_sv_dz
# Compute the squared deviation from Center of Mass along the range dimension
squared_dev = (sv_sel["range"] - CenterofMass) ** 2

# Multiply by Sv
sv_times_squared_dev = sv_sel * squared_dev

# Multiply by Delta_R to integrate along the range
sv_times_squared_dev_dz = sv_times_squared_dev * Delta_R

# Sum along range and normalize by Integrate_sv_dz
Inertia = sv_times_squared_dev_dz.sum(dim="range") / Integrate_sv_dz

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time,Inertia)
ax.set_title('Inertia')
ax.set_xlabel('Ping Time (d HH:MM)')
fig.savefig('Inertia.png', dpi=150)
plt.close(fig)

end = time.time()
print(f"Runtime: {end - start:.2f} seconds")
