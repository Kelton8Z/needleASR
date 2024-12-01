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

device = ndl.cuda()

# import wandb
# wandb.login(key=os.environ['WANDB_KEY'])

dropout = 0.1
batch_size = 1
seq_len = 5
input_dim = 40
hidden_size = 64
num_layers = 1
num_head = 2
dim_head = 16
causal = False

char_tokenizer = CharTokenizer()
vocab = char_tokenizer.vocab # a dictionary mapping characters to integers
inv_vocab = char_tokenizer.inv_vocab # a dictionary mapping integers to characters

LABELS = ARPAbet = list(vocab.keys())
OUT_SIZE = len(LABELS)

epochs = 50
train_config = {
    "beam_width" : 2,
    "epochs" : epochs,
    'batch_size' : batch_size,
    'learning_rate' : 0.001,
    'dropout_p' : dropout,
    'architecture' : 'transformer'                                    
}

# run = wandb.init(
#     name = "transformer with CTC", ## Wandb creates random run names if you skip this field
#     reinit = True, ### Allows reinitalizing runs when you re-run this cell
#     # run_id = ### Insert specific run id here if you want to resume a previous run
#     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
#     project = "needle-asr", ### Project should be created in your wandb account 
#     config = train_config ### Wandb Config for your run
# )

gc.collect()

dir = 'data/mini_librispeech_toy/'

print("Initialize train dataset")
train_data = ASRDataset(dir, "train")

print("Initialize val dataset")
val_data = ASRDataset(dir, "dev")

print("Initialize test dataset")
test_data = ASRDataset(dir, "test")

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = ndl.data.DataLoader(
    train_data, 
    batch_size=train_config['batch_size'], 
    collate_fn=train_data.collate_fn
)
val_loader = ndl.data.DataLoader(
    val_data, 
    batch_size=train_config['batch_size'], 
    shuffle=False, 
    collate_fn=val_data.collate_fn
)
test_loader = ndl.data.DataLoader(
    test_data, 
    batch_size=train_config['batch_size'],
    shuffle=False, 
    collate_fn=test_data.collate_fn
)

print("Batch size: ", train_config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(len(train_data), len(train_loader.dataset)))
print("Val dataset samples = {}, batches = {}".format(len(val_data), len(val_loader.dataset)))
print("Test dataset samples = {}, batches = {}".format(len(test_data), len(test_loader.dataset)))

# Sanity check: data loader
i= 0
for data in train_loader:
    x, y, lx, ly = data
    feat_seq_len = x.shape[1]
    print(x.shape, y.shape, lx.shape, ly.shape)
    print(f"feat_seq_len: {feat_seq_len}")
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
            sequence_len=sequence_len
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
    sequence_len=feat_seq_len, 
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
    
    # Get beam search results
    beam_results = generate(h, beam_width=train_config["beam_width"], blank_id=0, vocab=labels)
    
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

# Sanity check: model forward pass
for i, data in enumerate(train_loader):

    x, y, len_x, len_y = data
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"len_x shape: {len_x.shape}")
    print(f"len_y shape: {len_y.shape}")

    x, y, len_x, len_y = x.to(device), y.to(device), len_x.to(device), len_y.to(device)
    print(f"x device: {x.device}")
    print(f"y device: {y.device}")

    output = model(x)
    output = nn.ops.logsoftmax(output)

    print(f"output shape: {output.shape}")
    print(f"output type: {type(output)}")
    print(f"output device: {output.device}")
    
    print(f"y device: {y.device}")
    print(f"len_x device: {len_x.device}")
    print(f"len_y device: {len_y.device}")
    # print(f"output: {output}")
    print(f"y: {y}")
    print(f"len_x: {len_x}")
    print(f"len_y: {len_y}")
    # x_len_tup = tuple(map(int, len_x.detach().numpy().flatten()))
    # y_len_tup = tuple(map(int, len_y.detach().numpy().flatten()))
    t1 = time.time()
    torch_loss = torch_criterion(torch.tensor(output.transpose((0, 1)).detach().numpy()), torch.tensor(y.detach().numpy()), torch.tensor(len_x.detach().numpy(), dtype=torch.int32), torch.tensor(len_y.detach().numpy(), dtype=torch.int32))
    t2 = time.time()
    print(f"torch loss: {torch_loss}, after {t2-t1:.2f} seconds")

    t3 = time.time()
    loss = criterion(output, y, len_x, len_y)
    t4 = time.time()
    print(f"our loss: {loss}, after {t4-t3:.2f} seconds")

    loss.backward()

    if np.linalg.norm(torch_loss.detach().numpy() - loss.detach().numpy()) > 1e-4:
        print("Loss mismatch")
        break

    distance = calculate_levenshtein(output, y.detach().numpy(), len_x, len_y.detach().numpy(), LABELS, debug = True)
    print(f"lev-distance: {distance}")

    break


def evaluate(data_loader, model):
    val_dist = 0
    val_loss = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
    model.eval()
    for i, data in enumerate(data_loader):
        x, y, len_x, len_y = data
        x, y, len_x, len_y = x.to(device), y.to(device), len_x.to(device), len_y.to(device)
        output = model(x)

        loss = criterion(output, y, len_x, len_y)
        val_loss += loss

        batch_bar.set_postfix(
            loss = f"{loss/(i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()
          
        dist = calculate_levenshtein(output, y, len_x, len_y, LABELS, debug=False)
        val_dist += dist

        del x, y, len_x, len_y
            
    batch_bar.close()
    val_loss /= len(data_loader)
    val_dist /= len(data_loader)

    return val_loss, val_dist

# This is for checkpointing over multiple sessions

last_epoch_completed = 0
start = last_epoch_completed
end = train_config["epochs"]
best_val_dist = float("inf") # if restarting from some checkpoint, use that last number.
dist_freq = 1

def train_step(train_loader, model, optimizer, criterion):
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    train_loss = torch_train_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, len_x, len_y = data
        x, y, len_x, len_y = x.to(device), y.to(device), len_x.to(device), len_y.to(device)
        output = model(x)
        output = nn.ops.logsoftmax(output)
        loss = criterion(output, y, len_x, len_y)
        torch_loss = torch_criterion(torch.tensor(output.transpose((0, 1)).detach().numpy()), torch.tensor(y.detach().numpy()), torch.tensor(len_x.detach().numpy(), dtype=torch.int32), torch.tensor(len_y.detach().numpy(), dtype=torch.int32))

        loss.backward()
        optimizer.step()

        batch_bar.set_postfix(
            loss = f"{loss/(i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()

        train_loss += loss
        torch_train_loss += torch_loss
    
    batch_bar.close()
    train_loss /= len(train_loader) 
    torch_train_loss /= len(train_loader) 

    return train_loss, torch_train_loss

# The training loop
def train_asr(train_loader, val_loader, model, optimizer, criterion):
    for epoch in range(train_config["epochs"]):

        # one training step
        train_loss, torch_train_loss = train_step(train_loader, model, optimizer, criterion)
        writer.add_scalar('Loss/PyTorch', torch_loss.item(), epoch)
        writer.add_scalar('Loss/Needle', loss.item(), epoch)
        
        # Log the difference between losses
        loss_diff = abs(torch_loss.item() - loss.item())
        writer.add_scalar('Loss/Difference', loss_diff, epoch)
        
        # one validation step (to fail early as a test)
        val_loss, val_dist = evaluate(val_loader, model)
        # Calculating levenshtein distance isn't needed every epoch in the training step 
        print(f"val_loss: {val_loss}, val_dist: {val_dist}")
        # You may want to log some hyperparameters and results on wandb
        # wandb.log({"validation loss": val_loss, "validation distance": val_dist})
        
scheduler = None
train_asr(train_loader, val_loader, model, optimizer, criterion)
writer.flush()
