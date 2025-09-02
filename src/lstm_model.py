import torch
import torch.nn as nn
        
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden    
    

    def generate(self, start_sequence, num_tokens, device='cuda'):
        self.eval()
        with torch.no_grad():
            hidden = None
            
            if isinstance(start_sequence, list):
                input_seq = torch.tensor(start_sequence).unsqueeze(0).to(device)
            else:
                if start_sequence.dim() == 1:
                    input_seq = start_sequence.unsqueeze(0).to(device)
                else:
                    input_seq = start_sequence.to(device)

            _, hidden = self(input_seq, hidden)

            current_input = input_seq
            
            generated = []
            for _ in range(num_tokens):
                
                output, hidden = self(current_input, hidden)
            
                output = output[:, -1, :]
                
                probabilities = torch.softmax(output, dim=-1)
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
                
                generated.append(next_token.item())
                
                current_input = next_token.view(1, 1)

            return generated