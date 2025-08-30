import torch
import torch.nn as nn
from tqdm import tqdm
from src.eval_lstm import evaluate_model


def lstm_train(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x_batch)
            
            output = output.reshape(-1, output.size(-1))
            y_batch = y_batch.reshape(-1)
            
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        val_loss, val_acc, rouge1, rouge2 = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%} | ROUGE-1 SCORE: {rouge1:.2%} | ROUGE-2 SCORE: {rouge2:.2%}")

