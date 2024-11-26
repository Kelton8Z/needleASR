import soundfile
from needle.data import Dataset, DataLoader
from needle import Tensor, NDArray
import os
import sys
import librosa
import numpy as np
sys.path.append('./python')

class CharTokenizer():
    def __init__(self):
        self.eos_token = "<EOS>" # end of sentence token
        self.pad_token = "<PAD>" # padding token
        self.unk_token = "<UNK>" # unknown token

        # Initialize vocabulary with uppercase alphabet characters, prime, and space
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        # Create an inverse mapping from IDs to characters for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Define token IDs for special tokens for easy access
        self.eos_token_id = self.vocab[self.eos_token]
        self.sos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, data):
        # Split input data into a list of characters for tokenization
        return [char for char in data]

    def encode(self, data):
        # Encode each character in data to its integer ID, using unk_token if character is not in vocab
        e = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        return e

    def decode(self, data):
        # Decode list of token IDs back to string by mapping each ID to its character
        return ''.join([self.inv_vocab.get(j) for j in data])

class ASRDataset(Dataset):
    def __init__(
            self, 
            base_dir, 
            dataset_type, 
            feat_type="fbank", 
            feat_dim=40
        ):
        self.dataset_dir = os.path.join(base_dir, dataset_type)
        self.dataset_type = dataset_type
        self.flac_list = self.get_flac_list()
        self.transcripts = self.get_transcripts()
        self.feat_type = feat_type
        self.feat_dim = feat_dim
        self.tokenizer = CharTokenizer()
    
    def __getitem__(self, index):
        flac_path = self.flac_list[index]
        y, sr = soundfile.read(flac_path)
        feat = self.get_feat(y, sr, self.feat_type, self.feat_dim)
        # TODO: 
        #   Don't forget to ".to(device)" in the train script, 
        #   since here we don't know the device type
        feat = np.array(feat)
        feat = feat.transpose((0, 1)) # (T, feat_dim)

        key = flac_path.split('/')[-1].split('.')[0]
        transcript = self.transcripts[key]
        transcript_tokens = self.tokenizer.encode(transcript)
        transcript_tokens = np.array(transcript_tokens)

        # feat: np.ndarray (T, feat_dim), transcript_tokens: np.ndarray (L,)
        # NOTE: feat and transcript_tokens will be transformed to Tensor in dataloader __next__
        return feat, transcript_tokens 

    def __len__(self):
        assert len(self.flac_list) == len(self.transcripts), (
            f"Number of flac files {len(self.flac_list)} and "
            f"transcripts {len(self.transcripts)} do not match"
        )

        return len(self.flac_list)
    
    def get_flac_list(self):
        flac_list = []
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.flac'):
                    flac_list.append(os.path.join(root, file))
        
        with open(os.path.join(self.dataset_dir, 'flac_list.txt'), 'w') as f:
            for flac in flac_list:
                f.write(f'{flac}\n')
        
        return flac_list
    
    def get_transcripts(self):
        leaf_dir_list = []
        for root, subdirs, _ in os.walk(self.dataset_dir):
            if len(subdirs) == 0:
                # this is a leaf directory, which does not contain any subdirectories
                leaf_dir_list.append(root)
        
        transcripts = {}
        for dir in leaf_dir_list:
            dir_parts = dir.split('/')
            leaf_name = f"{dir_parts[-2]}-{dir_parts[-1]}"
            transcript_path = os.path.join(dir, f'{leaf_name}.trans.txt')

            assert os.path.exists(transcript_path), f"Transcript file {transcript_path} does not exist"

            with open(transcript_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]
        
        with open(os.path.join(self.dataset_dir, 'transcripts.txt'), 'w') as f:
            for key, val in transcripts.items():
                f.write(f'{key} {val}\n')
        
        return transcripts
    
    def get_feat(self, y, sr, feat_type, feat_dim):
        # use external library to extract features
        # use librosa
        if feat_type == "fbank":
            feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=feat_dim)
        elif feat_type == "mfcc":
            feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feat_dim)
        else:
            raise ValueError(f"Unsupported feature type: {feat_type}")
        
        return feat

    def collate_fn(batch):
        # pad feats in batch to same length
        # NOTE: there is no need to pad transcripts, since we use CTC-based ASR, 
        # where only exists encoder, no language model decoder based on transcripts. 
        # transcripts only serve as targets for CTC loss. 
        
        # the feat and transcript_tokens in batch input are all np.ndarray
        max_len = max([len(feat) for feat, _ in batch])
        batch_feats = [x[0] for x in batch]
        feats_lengths = [x.shape[0] for x in batch_feats]
        feats_lengths = np.array(feats_lengths) # (B, )
        padded_feats = np.zeros((len(batch), max_len, batch_feats[0].shape[1]))

        for i, feat in enumerate(batch_feats):
            padded_feats[i, :len(feats_lengths[i])] = feat
        
        padded_feats = Tensor(padded_feats) # TODO: .to(device) in train script
        feats_lengths = Tensor(feats_lengths, requires_grad=False)

        trancript_tokens = Tensor(np.array([x[1] for x in batch]), requires_grad=False) # (B, L)
        transcript_lengths = Tensor(np.array([len(x) for x in trancript_tokens]), requires_grad=False) # (B, )

        # padded_feats: Tensor (B, T, feat_dim), trancript_tokens: Tensor (B, L)
        # feats_lengths: Tensor (B, ), transcript_lengths: Tensor (B, )
        return padded_feats, trancript_tokens, feats_lengths, transcript_lengths





