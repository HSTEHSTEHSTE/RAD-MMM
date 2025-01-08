import pytorch_lightning as pl
import sys
import re
import yaml
sys.path.append('vocoders')
from pytorch_lightning.cli import LightningCLI
from tts_lightning_modules import TTSModel
from data_modules import BaseAudioDataModule
from jsonargparse import lazy_instance
from decoders import RADMMMFlow
from loss import RADTTSLoss
import inspect
from pytorch_lightning.callbacks import ModelCheckpoint
from training_callbacks import LogDecoderSamplesCallback, \
    LogAttributeSamplesCallback
from utils import get_class_args
from tts_text_processing.text_processing import TextProcessing
from common import Encoder
import torch

# radmmm_model_path = "/home/hltcoe/xli/ARTS/RAD-MMM/exp/decoder-a/latest-epoch_33-iter_199999.ckpt"
radmmm_model_path = "/home/hltcoe/xli/ARTS/RAD-MMM/exp/decoder-k/latest-epoch_5-iter_84999.ckpt"
model_config_paths = [
    # "configs/RADMMM_model_config.yaml",
    "configs/RADMMM_model_config_knnvc.yaml",
    "configs/RADMMM_f0model_config.yaml",
    "configs/RADMMM_energymodel_config.yaml",
    "configs/RADMMM_durationmodel_config.yaml",
    "configs/RADMMM_vpredmodel_config.yaml",
]
# data_config_path = "/home/hltcoe/xli/ARTS/RAD-MMM/configs/RADMMM_opensource_data_config_phonemizerless.yaml"
data_config_path = "/home/hltcoe/xli/ARTS/RAD-MMM/configs/RADMMM_opensource_data_config_phonemizerless_knnvc.yaml"
voc_model_path = "vocoders/hifigan_vocoder/g_00072000"
voc_config_path = "vocoders/hifigan_vocoder/config_16khz.json"
phonemizer_cfg='{"en_US": "assets/en_US_word_ipa_map.txt","es_MX": "assets/es_MX_word_ipa_map.txt","de_DE": "assets/de_DE_word_ipa_map.txt","en_UK": "assets/en_UK_word_ipa_map.txt","es_CO": "assets/es_CO_word_ipa_map.txt","es_ES": "assets/es_ES_word_ipa_map.txt","fr_FR": "assets/fr_FR_word_ipa_map.txt","hi_HI": "assets/hi_HI_word_ipa_map.txt","pt_BR": "assets/pt_BR_word_ipa_map.txt","te_TE": "assets/te_TE_word_ipa_map.txt"}'

# load the config
model_config = {
    'model': {}
}
for model_config_path in model_config_paths:
    with open(model_config_path, "r") as f:
        model_config['model'].update(yaml.safe_load(f)['model'])
with open(data_config_path, "r") as f:
    data_config = yaml.safe_load(f)

def instantiate_class(init):
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)

# instantiate submodules
model_config["model"]["add_bos_eos_to_text"] = False
model_config["model"]["append_space_to_text"] = True
model_config["model"]["decoder_path"] = radmmm_model_path
model_config["model"]["encoders_path"] = radmmm_model_path
model_config["model"]["handle_phoneme"] = "word"
model_config["model"]["handle_phoneme_ambiguous"] = "ignore"
model_config["model"]["heteronyms_path"] = "tts_text_processing/heteronyms"
model_config["model"]["output_directory"] = "tutorials/run1"
model_config["model"]["p_phoneme"] = 1
model_config["model"]["phoneme_dict_path"] = "tts_text_processing/cmudict-0.7b"
model_config["model"]["phonemizer_cfg"] = phonemizer_cfg
model_config["model"]["prediction_output_dir"] = "tutorials/out1"
model_config["model"]["prepend_space_to_text"] = True
model_config["model"]["sampling_rate"] = 16000
model_config["model"]["symbol_set"] = "radmmm_phonemizer_marker_segregated"
model_config["model"]["vocoder_checkpoint_path"] = voc_model_path
model_config["model"]["vocoder_config_path"] = voc_config_path

hparams = model_config["model"]
ttsmodel_kwargs={}
for k,v in hparams.items():
    if type(v) == dict and 'class_path' in v:
        print(k)
        ttsmodel_kwargs[k] = instantiate_class(v)
    elif k != "_instantiator":
        ttsmodel_kwargs[k] = v

# load the model from checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TTSModel.load_from_checkpoint(checkpoint_path=radmmm_model_path,\
                                      **ttsmodel_kwargs).to(device=device)

# initialize the datamodule
data_config["data"]["batch_size"]=1
data_config["data"]["phonemizer_cfg"]=phonemizer_cfg
data_config["data"]["inference_transcript"] = None 
data_module = BaseAudioDataModule(**data_config['data'])
data_module.setup(stage = "predict")

# run the input through the model
def run_inference(script, speaker_id, input_language_id, target_accent_id):
    inferData = [{
      "script": script,
      "spk_id": speaker_id,
      "decoder_spk_id": speaker_id,
      "duration_spk_id": speaker_id,
      "energy_spk_id": speaker_id,
      "f0_spk_id": speaker_id,
      "language": target_accent_id,
      "emotion": "other"
    }]
    
    ## set predictset
    data_module.predictset.combine_speaker_and_emotion = False
    data_module.predictset.data = inferData
    
    ## initialize and get the dataloader
    dl = data_module.predict_dataloader()
    
    ## get the first input
    inp = next(iter(dl))
    
    ## move the input tensors to GPU
    for k in inp.keys():
        if type(inp[k]) == torch.Tensor:
            inp[k] = inp[k].to(device=device)

    return model.forward(inp)

# fourth example - native german speaker speaking english in german accent
text = "What is the most resilient parasite"
# speaker_id = "mailabs-nadineeckert-other"
speaker_id = "ljs-00801-other"
input_language_id = "fr_FR"
target_accent_id = "en_US"

script = data_module.tp.convert_to_phoneme(text=text, phoneme_dict=data_module.tp.phonemizer_backend_dict[input_language_id])

print("Converted the sentence to phonemes: ", script)

output_file_path = run_inference(script=script, 
                                 speaker_id=speaker_id, 
                                 input_language_id=input_language_id, 
                                 target_accent_id=input_language_id)