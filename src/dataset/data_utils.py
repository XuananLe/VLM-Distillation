from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence


# Output [B, T_max, D]
def pad_sequence(sequences, padding_side='right', padding_value=0):
    return torch_pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
        padding_side=padding_side,
    )


# Per-sample processors keep a singleton batch axis; collation pads only the
# frame/image axis after removing that processor-local batch dimension.
def pad_frames(tensors, pad_value=0):
    return torch_pad_sequence(
        [frames.squeeze(0) for frames in tensors],
        batch_first=True,
        padding_value=pad_value,
    )
