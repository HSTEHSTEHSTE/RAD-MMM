import yaml
from data_modules import BaseAudioDataModule
from pathlib import Path
from tqdm import tqdm

phonemizer_cfg = '{"en_US": "assets/en_US_word_ipa_map.txt"}'

data_config_path = "/home/hltcoe/xli/ARTS/RAD-MMM/configs/RADMMM_opensource_data_config_phonemizerless_knnvc_l20.yaml"
with open(data_config_path, "r") as f:
    data_config = yaml.safe_load(f)

# initialize the datamodule
data_config["data"]["batch_size"] = 1
data_config["data"]["phonemizer_cfg"] = phonemizer_cfg
data_config["data"]["inference_transcript"] = None 
data_module = BaseAudioDataModule(**data_config['data'])
data_module.setup(stage = "predict")

input_language_id = "en_US"

file_lists = ['training_files', 'validation_files']
data = data_config['data']
target_filelist_basedir = 'multilingual-dataset/knnvc_libri_letters/filelists/'
orig_filelist_basedir = 'datasets/opensource/'
phone_filelist_basedir = 'multilingual-dataset/knnvc_libri/filelists/'
for file_list in file_lists:
    for file_split in data[file_list]:
        split_name = data[file_list][file_split]['basedir'].split('/')[-1]
        print(split_name)
        target_file_list = target_filelist_basedir + '/' + data[file_list][file_split]['filelist']
        Path(target_file_list).parents[0].mkdir(parents = True, exist_ok = True)
        file_list_file_dir = data[file_list][file_split]['filelist_basedir'] + '/' + data[file_list][file_split]['filelist']

        # Find orig file
        file_name = file_list_file_dir[len(phone_filelist_basedir):]
        file_name = file_name.replace('_phonemized', '')
        orig_file_name = orig_filelist_basedir + file_name
        files = {}
        with open(orig_file_name, 'r') as orig_file:
            for file_line in tqdm(orig_file):
                file_line_elements = file_line.strip().split('|')
                wav_name = file_line_elements[0]
                text = file_line_elements[1]
                # phones = data_module.tp.convert_to_phoneme(text = text, phoneme_dict = data_module.tp.phonemizer_backend_dict[input_language_id])
                phones = text
                files[wav_name] = phones

        with open(target_file_list, 'w') as target_file_list_file:
            with open(file_list_file_dir) as file_list_file:
                for file_line in tqdm(file_list_file):
                    file_line_elements = file_line.strip().split('|')
                    wav_name = '-'.join(file_line_elements[0][:-4].split('-')[:-1]) + '.wav'
                    if wav_name in files:
                        phones = files[wav_name]
                    target_file_line = file_line_elements[0] + '|' + phones + '|' + file_line_elements[2] + '|' + file_line_elements[3] + '|' + file_line_elements[4] + '\n'
                    target_file_list_file.write(target_file_line)
