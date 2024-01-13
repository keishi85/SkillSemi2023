class MetaDataRead:
    def __init__(self, file_path):
        self.file_path = file_path

    # Read mhd file to save dictionary
    def read_as_dict(self):
        data_dict = {}
        with open(self.file_path, 'r') as file:
            for line in file:
                if '=' in line:
                    key, value = line.strip().split('=')
                    data_dict[key.strip()] = value.strip()
        return data_dict