import os
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

def save_distping_parquet(dataout, output_path):
    nc_files = [os.path.join(dataout, 'sv', _F) for _F in os.listdir(os.path.join(dataout, 'sv')) if _F.endswith('.nc')]
    nc_files.sort()
    print('Reading nc files')

    variables = ['distance', 'raw_file', 'ping_time']

    data = []

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        var_data = [ds[var].values for var in variables]
        data.append(var_data)
        ds.close()

    table_data = {var: np.concatenate([var_data[i] for var_data in data]) for i, var in enumerate(variables)}

    df = pd.DataFrame(table_data)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    dataout = ""  # Provide the appropriate path here
    output_path = "distpingraw.parquet"
    save_distping_parquet(dataout, output_path)
