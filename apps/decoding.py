from typing import List
from collections import defaultdict

import sys
sys.path.append("python/")
import numpy as np
import torch
from needle import Tensor

def generate(logits, beam_width, blank_id, vocab: List[str] = None):
    """
    Implements beam search decoding for CTC-trained models.
    
    Args:
        logits: Network output Tensor of shape (time_steps, vocab_size)
        beam_width: Maximum number of beams to keep at each step
        blank_id: ID of the CTC blank token
        vocab: List of characters/tokens corresponding to model outputs
        
    Returns:
        List of tuples containing (decoded_sequence, log_probability)
        sorted by descending probability
    """
    import needle as ndl

    T, v = logits.shape
    lse = ndl.logsumexp(logits, 1).reshape((T, 1))
    log_probs = (logits - lse.broadcast_to(logits.shape)).numpy()
    
    # Initialize beam with empty sequence
    beam = {('', 0, -1): 0.0}  # (prefix, last_char, last_blank_pos): log_prob
    
    # Process each timestep
    for t in range(T):
        # top_k_probs, top_k_indices = torch.topk(Tensor(log_probs[t]), k=beam_width)
        top_k_indices = np.argpartition(log_probs[t], -beam_width)[-beam_width:]
        top_k_probs = log_probs[t][top_k_indices]

        # Sort them (since argpartition doesn't guarantee order)
        sorted_indices = np.argsort(-top_k_probs)  # - for descending order
        top_k_indices = top_k_indices[sorted_indices]
        top_k_probs = top_k_probs[sorted_indices]
                
        # Collect new candidates for beam， keys are tuples of (prefix, last_char, last_blank_pos)
        new_beam = defaultdict(float)
        
        for prob, c in zip(top_k_probs, top_k_indices):
            for (prefix, last_char, last_blank_pos), prefix_prob in beam.items():
                new_prob = prefix_prob + prob.item()
                
                if c == blank_id:
                    # Blank token - keep same prefix but update last_blank_pos
                    new_key = (prefix, last_char, t)
                    new_beam[new_key] = np.log(np.exp(new_beam[new_key]) + np.exp(new_prob)) if new_key in new_beam else new_prob
                    
                else:
                    # Skip repeated character with no blank in between
                    if c == last_char and last_blank_pos != t-1:
                        continue
                        
                    # Regular character - extend prefix
                    new_prefix = prefix + (vocab[c] if vocab else str(c))
                    new_key = (new_prefix, c, last_blank_pos)
                    new_beam[new_key] = np.log(np.exp(new_beam[new_key])+np.exp(new_prob)) if new_key in new_beam else new_prob
        
        
        new_beam[new_key] = np.log(np.exp(new_beam[new_key])+np.exp(new_prob)) if new_key in new_beam else new_prob
    
        # At each time-step, only the best scoring beams from the previous time-step are kept 
        beam = dict(
            sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )
    
    # Return final sequences and their probabilities
    return [(prefix, prob) for (prefix, _, _), prob in beam.items()]

def test_repeated_chars():
    """Test function to demonstrate correct handling of repeated characters"""
    # Create a simple vocabulary
    vocab = ['a', 'b', 'c', '-']  # '-' is blank
    
    # Create dummy logits that favor the sequence "abbc"
    # with blanks between the repeated 'b's
    T, V = 7, 4  # 7 timesteps: a-b-b-c
    logits = torch.full((T, V), -10.0)  # Make all probabilities very small
    
    # Set high probabilities for our desired sequence
    sequence = [0, 3, 1, 3, 1, 3, 2]  # a-b-b-c
    for t, s in enumerate(sequence):
        logits[t, s] = 0  # Make these tokens very likely
    
    results = generate(
        logits=Tensor(logits),
        beam_width=3,
        blank_id=3,
        vocab=vocab
    )
    
    print("\nTop decoded sequences:")
    for seq, log_prob in results:
        print(f"Sequence: {seq:10} Log-probability: {log_prob:.4f}")
        
if __name__ == "__main__":
    test_repeated_chars()
