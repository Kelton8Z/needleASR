import torch
# from torchsummaryX import summary
import sys

sys.path.append("python/")
import needle as ndl
import needle.nn as nn
from needle.data.datasets.librispeech_dataset import ASRDataset

import gc

from tqdm import tqdm
import os

# imports for decoding and distance calculation
# import ctcdecode
import Levenshtein

device = ndl.cpu()

import wandb
wandb.login(key=os.environ['WANDB_KEY'])

dropout = 0.1
batch_size = 8
seq_len = 5
input_dim = 27
hidden_size = 64
num_layers = 2
num_head = 8
dim_head = 32
causal = True

LABELS = ARPAbet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
OUT_SIZE = len(LABELS)

from models import LanguageModel
model = LanguageModel(embedding_size=input_dim, output_size=len(LABELS), hidden_size=hidden_size, num_layers=1,
                 seq_model='transformer', seq_len=40, device=device, dtype="float32")
'''
nn.Transformer(
    input_dim, hidden_size, num_layers,
    num_head=num_head,
    dim_head=dim_head,
    dropout=dropout,
    causal=causal,
    device=device,
    batch_first=True,
)'''

epochs = 50
train_config = {
    "beam_width" : 2,
    "epochs" : epochs,
    'batch_size' : batch_size,
    'learning_rate' : 0.001,
    'dropout_p' : dropout,
    'architecture' : 'transformer'                                    
}

run = wandb.init(
    name = "transformer with CTC", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "needle-asr", ### Project should be created in your wandb account 
    config = train_config ### Wandb Config for your run
)

from needle.nn.nn_ctcloss import CTCLoss
criterion = CTCLoss() #torch.nn.CTCLoss()# Define CTC loss as the criterion. losses are reduced
# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
optimizer =  ndl.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

# Declare the decoder. Use the CTC Beam Decoder to decode phonemes
# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
# from ctcdecode import CTCBeamDecoder
# decoder = CTCBeamDecoder(
#     labels=LABELS,
#     model_path=None,
#     alpha=0,
#     beta=0,
#     cutoff_top_n=40,
#     cutoff_prob=1.0,
#     beam_width=10,
#     num_processes=4,
#     blank_id=0,
#     log_probs_input=True
# )


# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode='min', factor = 0.5, patience = 4, threshold = 0.01)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * train_config["epochs"]))

# Mixed Precision, if needed
scaler = torch.cuda.amp.GradScaler()

# get me RAMMM!!!! 
import gc 
gc.collect()

dir = 'data/mini_librispeech_toy/'
train_data = ASRDataset(dir, "train") 

val_data = ASRDataset(dir, "dev") 

test_data = ASRDataset(dir, "test")

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = ndl.data.DataLoader(train_data, batch_size=train_config['batch_size'], collate_fn=train_data.collate_fn)

val_loader = ndl.data.DataLoader(val_data, batch_size=train_config['batch_size'],
                                         shuffle= False, collate_fn=val_data.collate_fn)

test_loader = ndl.data.DataLoader(test_data, batch_size=train_config['batch_size'],
                                         shuffle= False, collate_fn=test_data.collate_fn)


# def test_collate(batch):
#     batch_mfcc = batch

#     batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True) 
#     lengths_mfcc = [len(mfcc) for mfcc in batch_mfcc]

#     return batch_mfcc_pad, torch.tensor(lengths_mfcc)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'], pin_memory= True, 
#                                           shuffle= False, collate_fn=test_collate)

print("Batch size: ", train_config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(len(train_data), len(train_loader.dataset)))
print("Val dataset samples = {}, batches = {}".format(len(val_data), len(val_loader.dataset)))
print("Test dataset samples = {}, batches = {}".format(len(test_data), len(test_loader.dataset)))

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break 

torch.cuda.empty_cache()

from decoding import generate
import Levenshtein
def calculate_levenshtein(h, y, lh, ly, labels, debug = False):

    if debug:
        pass
        # print(f"\n----- IN LEVENSHTEIN -----\n")
        
    # As per docs for CTC.decoder, is returned here
    h = h.permute(1, 0, 2)
    beam_results, beam_scores, timesteps, out_lens = generate(h, beam_width=train_config["beam_width"], blank_id=0, vocab=labels) #decoder.decode(h, seq_lens = lh)
    #batch_size = beam_results.shape[0] # TODO
    batch_size = h.shape[0]
    distance = 0 # Initialize the distance to be 0 initially

    for i in range(batch_size): 
        #max_idx = torch.argmax(beam_scores[i])
        h_sliced = beam_results[i, 0, :out_lens[i, 0]] # [335]
        h_string = [ARPAbet[i] for i in h_sliced]
        h_string = "".join(h_string)

        y_sliced = y[i, :len_y[i]]
        y_string = [str(ARPAbet[i]) for i in y_sliced]
        y_string = "".join(y_string)

        distance += Levenshtein.distance(h_string, y_string)
    print("beam: ", beam_results[:, 0, :out_lens[:, 0]])
    print("mine: ", h_string)
    print("ref: ", y_string)
    distance /= batch_size # divide by batch size to get average distance

    return distance

with torch.no_grad():
  for i, data in enumerate(train_loader):
      
      #1. What values are you returning from the collate function
      #2. Move the features and target to <DEVICE>
      #3. Print the shapes of each to get a fair understanding 
      #4. Pass the inputs to the model
            # Think of the following before you implement:
            # 4.1 What will be the input to your model?
            # 4.2 What would the model output?
            # 4.3 Print the shapes of the output to get a fair understanding 

      # Calculate loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
      # Calculating the loss is not straightforward. Check the input format of each parameter
      x, y, len_x, len_y = data
      x, y = x.to(device), y.to(device)

      output, lengths_output = model(x, len_x)
      #input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=lengths_output, dtype=torch.long) 
      loss = criterion(output, y, lengths_output, len_y)
      #print(f"loss: {loss}")

      distance = calculate_levenshtein(output, y, lengths_output, ly, LABELS, debug = False)
      print(f"lev-distance: {distance}")

      break # one iteration is enough

torch.cuda.empty_cache()


def evaluate(data_loader, model):
    val_dist = 0
    val_loss = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
    model.eval()
    for i, data in enumerate(data_loader):
        x, y, len_x, len_y = data
        x, y = x.to(device), y.to(device)
        #with torch.cuda.amp.autocast():
        with torch.no_grad():
            output, lengths_output = model(x, len_x)
        #input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=lengths_output, dtype=torch.long) 

          #loss = criterion(output, y, lengths_output, len_y)
        # using mixed precision? 
        batch_bar.set_postfix(
            loss = f"{val_loss/(i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )

        #val_loss += loss
        
        batch_bar.update()
          
        dist = calculate_levenshtein(output, y, lengths_output, ly, LABELS, debug = False)
        val_dist += dist

        del x, y, len_x, len_y
        torch.cuda.empty_cache()

        break
            
    batch_bar.close()
    #val_loss /= len(data_loader)
    val_dist /= len(data_loader)

    return val_loss, val_dist

# This is for checkpointing over multiple sessions

last_epoch_completed = 0
start = last_epoch_completed
end = train_config["epochs"]
best_val_dist = float("inf") # if restarting from some checkpoint, use that last number.
dist_freq = 1

def train_step(train_loader, model, optimizer, criterion, scheduler, scaler):
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    train_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, len_x, len_y = data
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
          output, lengths_output = model(x, len_x)
          loss = criterion(output, y, lengths_output, len_y)

        # use mixed precision?
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss = f"{train_loss/(i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )

        # scheduler.step() # for cosine annealing
        train_loss += loss
        batch_bar.update()
    
    batch_bar.close()
    train_loss /= len(train_loader) 

    return train_loss 

torch.cuda.empty_cache()
gc.collect()

# The training loop
def train_asr(train_loader, val_loader, model, optimizer, criterion, scheduler, scaler):
    for epoch in range(train_config["epochs"]):

        # one training step
        train_loss = train_step(train_loader, model, optimizer, criterion, scheduler, scaler)
        # one validation step (to fail early as a test)
        val_loss, val_dist = evaluate(val_loader, model)
        # Calculating levenshtein distance isn't needed every epoch in the training step 

        # scheduler.step here for this particular scheduler
        if scheduler:
            scheduler.step(val_dist)
        
        # Use the below code to save models
        if val_dist < best_val_dist:
            #path = os.path.join(root_path, model_directory, 'checkpoint' + '.pth')
            print("Saving model")
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'val_dist': val_dist, 
                        'epoch': epoch}, './checkpoint.pth')
            best_val_dist = val_dist
            wandb.save('checkpoint.pth')
        

        # You may want to log some hyperparameters and results on wandb
        wandb.log({"validation loss": val_loss, "validation distance": val_dist})
        
scheduler = None
train_asr(train_loader, val_loader, model, optimizer, criterion, scheduler, scaler)

run.finish()
