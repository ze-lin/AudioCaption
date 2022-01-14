import sys
import os
from pathlib import Path
import multiprocessing
import json

import fire
from tqdm import tqdm, trange
import librosa
import numpy as np
import pandas as pd
import torch
import kaldiio

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.utils.train_util as train_util
import captioning.models.panns_inference_models as panns_inference_models
from panns_inference import labels
label2idx = {x:i for i,x in enumerate(labels)}

def load_model(config, checkpoint):
    dump = torch.load(checkpoint, map_location="cpu")
    encoder = getattr(
        captioning.models.encoder, config["encoder"])(
        config["data"]["raw_feat_dim"],
        config["data"]["fc_feat_dim"],
        config["data"]["attn_feat_dim"],
        **config["encoder_args"]
    )
    decoder = getattr(
        captioning.models.decoder, config["decoder"])(
        vocab_size=len(dump["vocabulary"]),
        **config["decoder_args"]
    )
    model = getattr(
        captioning.models, config["model"])(
        encoder, decoder, use_label=config["use_label"], **config["model_args"]
    )
    model.load_state_dict(dump["model"])
    model.eval()
    return model, dump["vocabulary"]

def load_feature_extractor(model_name, checkpoint):
    model = getattr(panns_inference_models, model_name)(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def load_audio(specifier: str):
    if specifier.endswith("|"):
        fd = kaldiio.utils.open_like_kaldi(specifier, "rb")
        mat = kaldiio.matio._load_mat(fd, None)
        fd.close()
        sr, y = mat
        y = y.copy() / 2 ** 15
    else:
        assert Path(specifier).exists(), specifier + " not exists!"
        y, sr = librosa.core.load(specifier, sr=None)
    if y.shape[0] == 0:
        return -1 
    y = librosa.core.resample(y, sr, 32000)
    return y


class InferenceLogMelDataset(torch.utils.data.Dataset):

    def __init__(self, aid_to_fname, aid_to_label):
        super().__init__()
        self.aid_to_fname = aid_to_fname
        self.aid_to_label = aid_to_label
        self.aids = list(aid_to_fname.keys())

    def __getitem__(self, idx):
        aid = self.aids[idx]
        label_text = self.aid_to_label[aid].split(";")
        label_idxs = [label2idx[x] for x in label_text]
        labels = torch.zeros(527).scatter_(0, torch.tensor(label_idxs), 1) 
        waveform = load_audio(self.aid_to_fname[aid])
        return aid, waveform, labels

    def __len__(self):
        return len(self.aids)


def collate_wrapper(min_length=0.32):
    def collate_fn(data_list):
        """
        data_list: [(aid1, waveform1), (aid2, waveform2), ...]
        """
        aids = []
        waveforms = []
        labels = []
        lengths = []
        blacklist_aids = []
        for item in data_list:
            waveform = item[1]
            audio_id = item[0]
            label = item[2]
            if isinstance(waveform, int) or len(waveform) < min_length * 32000:
                blacklist_aids.append(audio_id)
                continue
            aids.append(audio_id)
            waveforms.append(waveform)
            labels.append(label)
            lengths.append(waveform.shape[0])
        np_waveforms = np.zeros((len(waveforms), max(lengths)))
        for idx, waveform in enumerate(waveforms):
            np_waveforms[idx, :len(waveform)] = waveform
        return np.array(aids), np_waveforms, torch.stack(labels), np.array(lengths), blacklist_aids
    return collate_fn

def decode_caption(word_ids, vocabulary):
    candidate = []
    for word_id in word_ids:
        word = vocabulary[word_id]
        if word == "<end>":
            break
        elif word == "<start>":
            continue
        candidate.append(word)
    candidate = " ".join(candidate)
    return candidate

name_to_dr = {
    "Cnn10": 16,
    "Cnn14": 32,
    "Wavegram_Logmel_Cnn14": 32
}

def inference(input,
              output,
              label_dict,
              generator_checkpoint,
              generator_config,
              feature_extractor_name="Wavegram_Logmel_Cnn14",
              feature_extractor_checkpoint="sed/audioset_tagging_cnn/pretrained_weights/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
              batch_size=32,
              num_process=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    config = train_util.parse_config_or_kwargs(generator_config)
    model, vocabulary = load_model(config, generator_checkpoint)
    model = model.to(device)
    feature_extractor = load_feature_extractor(feature_extractor_name,
                                               feature_extractor_checkpoint)
    feature_extractor = feature_extractor.to(device)

    if Path(input).suffix == ".csv":
        wav_df = pd.read_csv(input, sep="\t")
        aid_to_fname = dict(zip(wav_df["audio_id"], wav_df["file_name"]))
    elif Path(input).suffix in [".wav", ".mp3", ".flac"]:
        audio_id = Path(input).name
        aid_to_fname = {audio_id: input}
    else:
        raise Exception("input should be wav.csv or audio file")

    label_df = pd.read_csv(label_dict, sep="\t")
    aid_to_label = dict(zip(label_df["audio_id"], label_df["event_labels"]))

    downsample_ratio = name_to_dr[feature_extractor_name]

    dataloader = torch.utils.data.DataLoader(
        InferenceLogMelDataset(aid_to_fname, aid_to_label),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper(0.01 * downsample_ratio),
        num_workers=num_process
    )

    captions = []
    audio_ids = []
    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            aids, waveforms, labels, lengths, _ = batch
            waveforms = torch.as_tensor(waveforms).float().to(device)
            labels = labels.to(device)
            audio_feats = feature_extractor(waveforms)
            audio_feat_lengths = np.floor((lengths / 320 + 1) / downsample_ratio)
            
            for sample_idx in range(audio_feat_lengths.shape[0]):
                if audio_feat_lengths[sample_idx] > audio_feats["attn_feat"][sample_idx].shape[0]:
                    audio_feat_lengths[sample_idx] = audio_feats["attn_feat"][sample_idx].shape[0]
            # import pdb; pdb.set_trace()
            input_dict = {
                "mode": "inference",
                "raw_feats": None,
                "raw_feat_lens": None,
                "fc_feats": audio_feats["fc_feat"],
                "attn_feats": audio_feats["attn_feat"],
                "attn_feat_lens": torch.as_tensor(audio_feat_lengths, dtype=torch.long),
                "sample_method": "beam",
                "labels": labels
            }
            output_dict = model(input_dict)
            caption_batch = [decode_caption(seq, vocabulary) for seq in output_dict["seqs"].cpu().numpy()]
            captions.extend(caption_batch)
            audio_ids.extend(aids)
            pbar.update()
    
    assert len(audio_ids) == len(captions)
    data = {aid: caption for aid, caption in zip(audio_ids, captions)}
    json.dump(data, open(output, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(inference)

