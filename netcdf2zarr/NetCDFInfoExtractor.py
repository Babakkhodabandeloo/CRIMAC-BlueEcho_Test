import os
import netCDF4
import csv
import sys

class NetCDFInfoExtractor:
    def __init__(self, folder_path, output_file):
        self.folder_path = folder_path
        self.output_file = output_file

    def extract_info(self):
        file_data = {}

        # Loop through all files in the folder
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.nc'):
                file_path = os.path.join(self.folder_path, filename)
                data = self.read_netcdf_file(file_path)
                data['Filesize (MB)'] = self.get_file_size_mb(file_path)
                file_data[filename] = data

        return file_data

    def read_netcdf_file(self, file_path):
        data = {'Coordinates': {}, 'Data variables': {}, 'channel_id': '', 'frequency': ''}

        with netCDF4.Dataset(file_path, 'r') as nc:
            for name, variable in nc.variables.items():
                info = f"{str(variable.dtype)}:{variable.size}"
                if name == 'channel_id' or name == 'frequency':
                    data[name] = self.format_array_to_string(variable[:])
                elif name in nc.dimensions:
                    data['Coordinates'][name] = info
                else:
                    data['Data variables'][name] = info

        return data

    @staticmethod
    def format_array_to_string(array):
        return ';'.join(map(str, array))

    @staticmethod
    def get_file_size_mb(file_path):
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)

    def write_to_csv(self, file_data):
        all_keys = set()
        for data in file_data.values():
            all_keys.update(data['Coordinates'].keys())
            all_keys.update(data['Data variables'].keys())
        header = ['Filename', 'Filesize (MB)', 'channel_id', 'frequency'] + sorted(all_keys)

        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for filename, data in file_data.items():
                row = {
                    'Filename': filename,
                    'Filesize (MB)': data['Filesize (MB)'],
                    'channel_id': data['channel_id'],
                    'frequency': data['frequency']
                }
                for key in all_keys:
                    row[key] = data['Coordinates'].get(key, '') or data['Data variables'].get(key, '')
                writer.writerow(row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python NetCDFInfoExtractor.py <folder_path> <output_csv_file>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2]

    extractor = NetCDFInfoExtractor(folder_path, output_file)
    file_data = extractor.extract_info()
    extractor.write_to_csv(file_data)
    print(f"NetCDF information extracted and saved to {output_file}")
