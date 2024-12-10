import gc
import os
import Levenshtein
import sys
sys.path.append("python/")

import needle as ndl
import needle.nn as nn
import numpy as np

from tqdm import tqdm
from needle.data.datasets.librispeech_dataset import ASRDataset
from needle.nn.nn_ctcloss import CTCLoss
from decoding import generate
from needle.data.datasets.librispeech_dataset import CharTokenizer
from matplotlib import pyplot as plt

device = ndl.cpu()

# import wandb
# wandb.login(key=os.environ['WANDB_KEY'])

dropout = 0.1
batch_size = 1
input_dim = 2
hidden_size = 4
num_layers = 1
num_head = 2
dim_head = 4
causal = False
dataset_trunc_train = 1
dataset_trunc_dev = 1

char_tokenizer = CharTokenizer()
vocab = char_tokenizer.vocab # a dictionary mapping characters to integers
inv_vocab = char_tokenizer.inv_vocab # a dictionary mapping integers to characters

LABELS = ARPAbet = list(vocab.keys())
OUT_SIZE = len(LABELS)
DEBUG = False

epochs = 2
train_config = {
    "beam_width" : 1,
    "epochs" : epochs,
    'batch_size' : batch_size,
    'learning_rate' : 1e-5,
    'dropout_p' : dropout,
    'architecture' : 'transformer'                                    
}

gc.collect()

dir = 'data/mini_librispeech_toy/'

print("Initialize train dataset")
train_data = ASRDataset(dir, "train", feat_dim=input_dim, trunc=dataset_trunc_train, max_len_feat=341, max_len_transcript=149)

print("Initialize val dataset")
val_data = ASRDataset(dir, "dev", feat_dim=input_dim, trunc=dataset_trunc_dev, max_len_feat=341, max_len_transcript=149)

print("Initialize test dataset")
test_data = ASRDataset(dir, "test", feat_dim=input_dim, trunc=dataset_trunc_dev, max_len_feat=341, max_len_transcript=149)

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = ndl.data.DataLoader(
    train_data, 
    batch_size=train_config['batch_size'], 
    shuffle=False, 
    # collate_fn=train_data.collate_fn if train_config['batch_size'] > 1 else train_data.collate_fn_batch_1
    collate_fn=train_data.collate_fn
)
val_loader = ndl.data.DataLoader(
    val_data, 
    batch_size=train_config['batch_size'], 
    shuffle=False, 
    collate_fn=train_data.collate_fn
)
test_loader = ndl.data.DataLoader(
    test_data, 
    batch_size=train_config['batch_size'],
    shuffle=False, 
    collate_fn=train_data.collate_fn
)

# Sanity check: data loader
print(f"Sanity check: data loader")
i= 0
for data in train_loader:
    x, y, lx, ly = data
    feat_seq_len_train = x.shape[1]
    print("Train data")
    print(f"x shape: {x.shape} y shape: {y.shape} lx shape: {lx.shape} ly shape: {ly.shape}")
    print(f"feat_seq_len: {feat_seq_len_train}")
    i += 1
    if i == 2:
        break 
for data in val_loader:
    x, y, lx, ly = data
    feat_seq_len_val = x.shape[1]
    print("Val data")
    print(f"x shape: {x.shape} y shape: {y.shape} lx shape: {lx.shape} ly shape: {ly.shape}")
    print(f"feat_seq_len: {feat_seq_len_val}")
    i += 1
    if i == 2:
        break
for data in test_loader:
    x, y, lx, ly = data
    feat_seq_len_test = x.shape[1]
    print("Test data")
    print(f"x shape: {x.shape} y shape: {y.shape} lx shape: {lx.shape} ly shape: {ly.shape}")
    print(f"feat_seq_len: {feat_seq_len_test}")
    i += 1
    if i == 2:
        break

class ASRModel(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        hidden_size, 
        num_layers, 
        num_head, 
        dim_head, 
        dropout, 
        causal, 
        device, 
        dtype, 
        batch_first, 
        sequence_len, 
        if_positional_embedding, 
        vocal_size=OUT_SIZE
    ):
        self.encoder = nn.Transformer(
            embedding_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            num_head=num_head, 
            dim_head=dim_head, 
            dropout=dropout, 
            causal=causal, 
            device=device, 
            dtype=dtype, 
            batch_first=batch_first, 
            sequence_len=sequence_len, 
            if_positional_embedding=if_positional_embedding
        )
        self.linear = nn.Linear(input_dim, vocal_size, device=device, dtype=dtype)
        
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.linear(x)
        return x


model = ASRModel(
    embedding_size=input_dim, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    num_head=num_head, 
    dim_head=dim_head, 
    dropout=dropout, 
    causal=causal, 
    device=device, 
    dtype="float32", 
    batch_first=True, 
    sequence_len=feat_seq_len_train, 
    if_positional_embedding=True, 
    vocal_size=OUT_SIZE
)

criterion = CTCLoss(batch_first=True)
import torch
torch_criterion = torch.nn.CTCLoss()
optimizer =  ndl.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

def calculate_levenshtein(h, y, lh, ly, labels, debug=False):
    """
    Calculate average Levenshtein distance between beam search results and reference sequences.
    
    Args:
        h: Input logits tensor of shape (batch_size, time_steps, vocab_size)
        y: Reference sequences tensor
        lh: Lengths of input sequences
        ly: Lengths of reference sequences
        labels: Vocabulary labels (ARPAbet in this case)
        debug: Whether to print debug information
        
    Returns:
        Average Levenshtein distance across the batch
    """
    if debug:
        print(f'h shape {h.shape}')

    h, y, lh, ly = h.numpy(), y.numpy(), lh.numpy(), ly.numpy()
    
    # Get beam search results
    beam_results = generate(h, beam_width=train_config["beam_width"], blank_id=0, vocab=labels)
    print(f"decode results: {beam_results[0][0]}")
    
    batch_size = h.shape[0]
    distance = 0
    
    for i in range(batch_size):
        # Get the best hypothesis (first result) for this sequence
        if beam_results[i]:  # Check if we got any results
            h_string = beam_results[i][0][0]  # Take the string from the first (best) result tuple
        else:
            h_string = ""  # Empty string if no results
            
        # Process reference sequence
        y_sliced = y[i, :int(ly[i])]
        y_string = "".join(str(ARPAbet[int(j)]) for j in y_sliced)
        
        # Calculate Levenshtein distance
        distance += Levenshtein.distance(h_string, y_string)
        
        if debug:
            print("beam result:", h_string)
            print("reference:", y_string)
    
    # Calculate average distance
    average_distance = distance / batch_size
    
    return average_distance

import datetime
import time
# Initialize TensorBoard writer
from torch.utils.tensorboard import SummaryWriter
log_dir = f"runs/training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)


def evaluate(data_loader, model):
    val_dist = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=True, position=0, desc='Val') 
    model.eval()
    for i, data in enumerate(data_loader):
        x, y, len_x, len_y = data
        x, y, len_x, len_y = x.to(device), y.to(device), len_x.to(device), len_y.to(device)
        output = model(x)
          
        dist = calculate_levenshtein(output, y, len_x, len_y, LABELS, debug=False)
        batch_bar.set_postfix(
            dist = f"{dist.numpy()/(i+1):.4f}"
        )
        batch_bar.update()

        val_dist += dist

        del x, y, len_x, len_y
            
    batch_bar.close()
    val_dist /= len(data_loader)

    return val_dist

# This is for checkpointing over multiple sessions

last_epoch_completed = 0
start = last_epoch_completed
end = train_config["epochs"]
best_val_dist = float("inf") # if restarting from some checkpoint, use that last number.
dist_freq = 1

def train_step(train_loader, model, optimizer, criterion, epoch):
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=True, position=0, desc='Train') 
    print('\n')
    train_loss = torch_train_loss = 0
    train_loss_steps = []
    torch_train_loss_steps = []

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, len_x, len_y = data
        x, y, len_x, len_y = x.to(device), y.to(device), len_x.to(device), len_y.to(device)

        output = model(x)
        output = nn.ops.logsoftmax(output)

        if DEBUG:
            parameters = model.parameters()
            print(f"parameters: {parameters}")

        
        loss = criterion(output, y, len_x, len_y)
        torch_loss = torch_criterion(
            torch.tensor(output.transpose((0, 1)).numpy(), dtype=torch.float32), 
            torch.tensor(y.numpy(), dtype=torch.float32), 
            torch.tensor(len_x.numpy(), dtype=torch.int32), 
            torch.tensor(len_y.numpy(), dtype=torch.int32)
        )
        train_loss_steps.append(loss.numpy())
        torch_train_loss_steps.append(torch_loss.item())

        writer.add_scalar('Loss (Step)/Needle', loss.numpy(), epoch * len(train_loader) + i)
        writer.add_scalar('Loss (Step)/PyTorch', torch_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalars('Losses (Step)', {'Needle': loss.numpy(), 'PyTorch': torch_loss.item()}, epoch * len(train_loader) + i)
        writer.flush()

        loss.backward()
        optimizer.step()

        if DEBUG:
            parameters_updated = model.parameters()
            print(f"parameters updated: {parameters_updated}")

        batch_bar.set_postfix(
            our_loss=f"{loss.numpy()/(i+1):.4f}",
            torch_loss=f"{torch_loss.item()/(i+1):.4f}"
        )
        batch_bar.update()

        train_loss += loss
        torch_train_loss += torch_loss
    
    plot_loss(train_loss_steps, torch_train_loss_steps, f"Step @ Epoch {epoch}", epoch)

    batch_bar.close()
    train_loss /= len(train_loader) 
    torch_train_loss /= len(train_loader)


    return train_loss, torch_train_loss

def plot_loss(train_loss_list, train_torch_loss_list, xlabel, epoch=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label="Needle loss")
    plt.plot(train_torch_loss_list, label="PyTorch loss")
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    if epoch == None:
        plt.savefig(f"train_loss_total.png")
    else:
        plt.savefig(f"train_loss_ep{epoch}.png")
    plt.close()

def plot_dist(val_dist_list, xlabel="Epoch"):
    plt.figure(figsize=(10, 6))
    plt.plot(val_dist_list, label="Levenshtein distance")
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Distance")
    plt.title("Validation distance")
    plt.legend()
    plt.savefig(f"val_dist.png")

def plot_val_loss(val_loss_list, xlabel="Epoch"):
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_list, label="Validation loss")
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.legend()
    plt.savefig(f"val_loss.png")

# The training loop
def train_asr(train_loader, val_loader, model, optimizer, criterion):
    train_loss_list = []
    train_torch_loss_list = []
    val_dist_list = []
    val_loss_list = []

    for epoch in range(train_config["epochs"]):
        # one training step
        train_loss, torch_train_loss = train_step(train_loader, model, optimizer, criterion, epoch)
        train_loss_list.append(train_loss.numpy())
        train_torch_loss_list.append(torch_train_loss.item())
        writer.add_scalar('Loss (Epoch)/NeedleTrain', train_loss.numpy(), epoch)
        writer.add_scalar('Loss (Epoch)/PyTorchTrain', torch_train_loss.item(), epoch)
        writer.add_scalars('Losses (Epoch)', {'NeedleTrain': train_loss.numpy(), 'PyTorchTrain': torch_train_loss.item()}, epoch)
        writer.flush()
        
        # one validation step (to fail early as a test)
        val_dist = evaluate(val_loader, model)
        val_dist_list.append(val_dist.item())
        writer.add_scalar('Distance (Epoch)/Val', val_dist.item(), epoch)

    plot_loss(train_loss_list, train_torch_loss_list, "Epoch")
    # plot_dist(val_dist_list)
    # plot_val_loss(val_loss_list)
        
scheduler = None
train_asr(train_loader, val_loader, model, optimizer, criterion)
writer.flush()
