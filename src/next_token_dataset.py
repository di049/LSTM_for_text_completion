from torch.utils.data import Dataset
import torch

class TweetsDataset(Dataset):
    def __init__(self, texts, seq_len=128, pad_token=50256):
        self.samples = []
        for line in texts:
            if len(line) < seq_len:
                line = line + [pad_token] * (seq_len - len(line))
            else:
                line = line[:seq_len]

            input = torch.tensor(line[:-1], dtype=torch.long)
            target = torch.tensor(line[1:], dtype=torch.long)

            self.samples.append((input, target))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
