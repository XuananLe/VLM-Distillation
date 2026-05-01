from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
# Output [B, T_max, D]
def pad_sequence(sequences, padding_side='right', padding_value=0):
    return torch_pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
        padding_side=padding_side,
    )
