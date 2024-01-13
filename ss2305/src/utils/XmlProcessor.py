import os
import shutil
import xml.etree.ElementTree as ET


class XmlProcessor:
    def __init__(self):
        self.input_dir_name = None

    # Input directory name and output directory
    def load_xml(self, input_dir_name):
        self.input_dir_name = input_dir_name
        # check if input directory exists
        try:
            if not os.path.exists(input_dir_name):
                raise FileNotFoundError(f'cannot found {input_dir_name}')
        except FileNotFoundError as e:
            print(f'Error: {e}')

        # Find all .xml files in specified directory
        # xml_files = glob.glob(os.path.join(input_dir_name, '*.xml'))
        xml_file_list = []
        for subdir, dir, files in os.walk(input_dir_name):
            for file in files:
                if file.endswith('.xml'):
                    full_path = os.path.join(subdir, file)
                    xml_file_list.append(full_path)
        return xml_file_list

    # Rewrite "PatientName" and "PatientBirthDate" to "*"
    # Make directory if output directory does not exist
    def anonymize_xml(self, directory_path, output_dir_name):
        self.copy_dir(directory_path, output_dir_name)
        xml_files_list = self.load_xml(output_dir_name)

        if not os.path.exists(output_dir_name):
            full_output_path = os.path.join(self.input_dir_name, output_dir_name)
            os.makedirs(full_output_path)

        for xml_file in xml_files_list:
            parent_dir = os.path.dirname(xml_file)
            output_dir_path = os.path.join(output_dir_name, parent_dir)
            output_dir_path = os.path.join(output_dir_path, xml_file.split('/')[-1])
            self.process_file(xml_file, output_dir_path)

    def process_file(self, xml_file, output_dir_path):
        # Tags to anonymize
        tags_to_anonymize = ['PatientName', 'PatientBirthDate']
        # Read xml file
        print(f'xml path : {xml_file}')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find and replace tags that need to be anonymized
        for tag in tags_to_anonymize:
            for element in root.iter(tag):
                element.text = '*'

        # Preserve xml file
        tree.write(output_dir_path, encoding="UTF-8", xml_declaration=True, method="xml")

        newFile = open("temp.txt", "w", encoding='utf8')
        with open(output_dir_path, encoding='utf8') as file:
            for line in file:
                if line.strip().startswith("<?xml"):
                    newFile.write(line.replace("'", '"'))
                else:
                    newFile.write(line.replace(' />', '/>'))
            newFile.write('\n')
        newFile.close()

        #1 delete orignal XML
        #2 Rename temp.txt
        os.remove(output_dir_path)
        os.rename("temp.txt", output_dir_path)

    def copy_dir(self, input_path, output_path):
        for item in os.listdir(input_path):
            s = os.path.join(input_path, item)
            d = os.path.join(output_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)


if __name__ == '__main__':
    dir_path = '../../data/list'
    preserve_path = '../../data/anonymized_list'

    xml = XmlProcessor()
    xml.anonymize_xml(dir_path, preserve_path)






