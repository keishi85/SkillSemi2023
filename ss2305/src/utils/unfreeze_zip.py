import os
import zipfile


def unfreeze_zip(root_dir_path, destination_path):
    # Search all directory
    for subdir, dirs, files in os.walk(root_dir_path):
        for file in files:
            if file.endswith('.zip'):
                # full path of zip file
                zip_path = os.path.join(subdir, file)

                # Extract preservation name
                extract_folder_name = os.path.splitext(file)[0]
                extract_path = os.path.join(destination_path, extract_folder_name)

                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)

                # Open zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Unfreeze zip file
                    zip_ref.extractall(extract_path)






if __name__ == '__main__':
    unfreeze_zip('../../data/anonymized_list', '../../data/anonymized_list')