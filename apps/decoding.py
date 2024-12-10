from typing import List, Tuple
from collections import defaultdict

import sys
sys.path.append("python/")
import numpy as np
import torch
import needle as ndl
from needle import Tensor
from pprint import pprint

def generate(logits, beam_width: int, blank_id: int, vocab: List[str] = None) -> List[List[Tuple[str, float]]]:
    """
    Implements beam search decoding for CTC-trained models with batch processing.
    
    Args:
        logits: Network output Tensor of shape (batch_size, time_steps, vocab_size)
        beam_width: Maximum number of beams to keep at each step
        blank_id: ID of the CTC blank token
        vocab: List of characters/tokens corresponding to model outputs
        
    Returns:
        List of lists, where each inner list contains tuples of (decoded_sequence, log_probability)
        for each sequence in the batch, sorted by descending probability
    """
    B, T, V = logits.shape  # batch_size, time_steps, vocab_size
    
    # Convert logits to log probabilities
    lse = ndl.logsumexp(logits, 2).reshape((B, T, 1))
    log_probs = (logits - lse.broadcast_to(logits.shape)).numpy()
    
    # Initialize batch of beams
    batch_beams = []
    for b in range(B):
        # Initialize each sequence with empty beam
        batch_beams.append({('', 0, -1): 0.0})  # (prefix, last_char, last_blank_pos): log_prob
    
    # Process each sequence in the batch
    results = []
    for b in range(B):
        beam = batch_beams[b]
        sequence_log_probs = log_probs[b]  # (time_steps, vocab_size)
        
        # Process each timestep
        for t in range(T):
            # Get top-k probabilities and indices for current timestep
            top_k_indices = np.argpartition(sequence_log_probs[t], -beam_width)[-beam_width:]
            top_k_probs = sequence_log_probs[t][top_k_indices]
            
            # Sort them (since argpartition doesn't guarantee order)
            sorted_indices = np.argsort(-top_k_probs)  # - for descending order
            top_k_indices = top_k_indices[sorted_indices]
            top_k_probs = top_k_probs[sorted_indices]
            
            # Collect new candidates for beam
            new_beam = defaultdict(float)
            
            for prob, c in zip(top_k_probs, top_k_indices):
                for (prefix, last_char, last_blank_pos), prefix_prob in beam.items():
                    new_prob = prefix_prob + prob.item()
                    
                    if c == blank_id:
                        # Blank token - keep same prefix but update last_blank_pos
                        new_key = (prefix, last_char, t)
                        new_beam[new_key] = np.log(np.exp(new_beam[new_key]) + np.exp(new_prob)) \
                            if new_key in new_beam else new_prob
                    else:
                        # Skip repeated character with no blank in between
                        if c == last_char and last_blank_pos != t-1:
                            continue
                            
                        # Regular character - extend prefix
                        new_prefix = prefix + (vocab[c] if vocab else str(c))
                        new_key = (new_prefix, c, last_blank_pos)
                        new_beam[new_key] = np.log(np.exp(new_beam[new_key]) + np.exp(new_prob)) \
                            if new_key in new_beam else new_prob
            
            # Keep only the best scoring beams
            beam = dict(sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width])
        
        # Store results for this sequence
        results.append([(prefix, prob) for (prefix, _, _), prob in beam.items()])
    
    return results


def test_repeated_chars():
    """Test function to demonstrate correct handling of repeated characters in batched mode"""
    
    # Create a simple vocabulary
    vocab = ['a', 'b', 'c', '-']  # '-' is blank
    print("Vocabulary:", vocab)
    
    # Create dummy logits that favor the sequence "abbc"
    # with blanks between the repeated 'b's
    B, T, V = 2, 7, 4  # batch_size=2, 7 timesteps: a-b-b-c, 4 vocab items
    logits = np.full((B, T, V), -10.0)  # Make all probabilities very small
    
    # Set high probabilities for our desired sequences
    # Sequence 1: "abbc"
    sequence1 = [0, 3, 1, 3, 1, 3, 2]  # a-b-b-c
    for t, s in enumerate(sequence1):
        logits[0, t, s] = 0  # Make these tokens very likely
    
    # Sequence 2: "abc" (different sequence for batch diversity)
    sequence2 = [0, 3, 1, 3, 3, 3, 2]  # a-b--c
    for t, s in enumerate(sequence2):
        logits[1, t, s] = 0  # Make these tokens very likely
    
    results = generate(
        logits=ndl.Tensor(logits),
        beam_width=3,
        blank_id=3,
        vocab=vocab
    )
    
    print("Beam width: 3")
    for batch_idx, batch_results in enumerate(results):
        print(f"\nSequence {batch_idx + 1}:")
        print("Input sequence:", "".join([vocab[s] for s in (sequence1 if batch_idx == 0 else sequence2)]))
        print("Beam search results:")
        for seq, log_prob in batch_results:
            print(f"Decoded sequence: '{seq}' \nLog probability: {log_prob:.4f}")
        
if __name__ == "__main__":
    test_repeated_chars()
