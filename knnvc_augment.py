import os
from os.path import join, isfile
from pathlib import Path
import torch, torchaudio
import torch.nn.functional as F
import yaml
from tqdm import tqdm

device = 'cuda'
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
knn_vc.to(device)


target_spk_paths = {
    '10114': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb1/dev/wav/id10114', # Bruno Ganz
    '10092': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb1/dev/wav/id10092', # Bingbing Li
    '09171': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id09171', # Zendaya
    '01493': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id01493', # Cesc Fàbregas
    '09185': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id09185', # Zlatan Ibrahimović
    '05272': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id05272', # Louis van Gaal
    '02262': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id00262', # Alex Morgan
    '05351': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id05351', # Luke Shaw
    '00801': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id00801', # Asamoah Gyan
    '08327': '/home/hltcoe/xli/ARTS/Voice-Privacy-Challenge-2024/corpora/voxceleb/voxceleb2/dev/wav/id08327', # Susie Wolff
}

matching_sets = {}
for target_spk in tqdm(target_spk_paths):
    target_spk_wavs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(target_spk_paths[target_spk]) for f in filenames if os.path.splitext(f)[1] == '.wav']
    matching_sets[target_spk] = None
    # matching_sets[target_spk] = knn_vc.get_matching_set(target_spk_wavs)

data_config_file = '/home/hltcoe/xli/ARTS/RAD-MMM/configs/RADMMM_opensource_data_config_phonemizerless.yaml'
target_rootdir = 'multilingual-dataset/knnvc/opensource/'
target_filelist_basedir = 'multilingual-dataset/knnvc/filelists/'
with open(data_config_file) as data_config:
    data = yaml.safe_load(data_config)['data']
file_lists = ['training_files', 'validation_files']
for file_list in file_lists:
    for file_split in data[file_list]:
        split_name = data[file_list][file_split]['basedir'].split('/')[-1]
        print(split_name)
        target_wavs_dir = target_rootdir + '/' + split_name + '/' + data[file_list][file_split]['sampling_rate']
        Path(target_wavs_dir).mkdir(parents = True, exist_ok = True)
        wavs_dir = data[file_list][file_split]['basedir'] + '/' + data[file_list][file_split]['sampling_rate']
        target_file_list = target_filelist_basedir + '/' + data[file_list][file_split]['filelist']
        Path(target_file_list).parents[0].mkdir(parents = True, exist_ok = True)
        file_list_file_dir = data[file_list][file_split]['filelist_basedir'] + '/' + data[file_list][file_split]['filelist']
        with open(target_file_list, 'w') as target_file_list_file:
            with open(file_list_file_dir) as file_list_file:
                for file_line in tqdm(file_list_file):
                    file_line_elements = file_line.strip().split('|')
                    wav_name = file_line_elements[0]
                    wav_dir = wavs_dir + '/' + wav_name
                    # query_seq = knn_vc.get_features(wav_dir)
                    for target_spk_id in matching_sets:
                        target_wav_name = wav_name.split('.')[0] + '-' + target_spk_id + '.wav'
                        
                        if not os.path.isfile(target_wavs_dir + '/' + target_wav_name):
                            out_wav = knn_vc.match(query_seq, matching_sets[target_spk_id])
                            Path(target_wavs_dir + '/' + target_wav_name).parents[0].mkdir(parents = True, exist_ok = True)
                            torchaudio.save(target_wavs_dir + '/' + target_wav_name, out_wav.unsqueeze(0), sample_rate = 16000)
                        else:
                            out_wav = torchaudio.load(target_wavs_dir + '/' + target_wav_name)[0].squeeze(0)
                        target_file_line = target_wav_name + '|' + file_line_elements[1] + '|' + file_line_elements[2] + '|' + file_line_elements[3] + '|' + str(out_wav.shape[0] / 16000)
                        target_file_line += '\n'
                        target_file_list_file.write(target_file_line)