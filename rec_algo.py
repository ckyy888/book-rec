import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class BookRecommenderDataset(Dataset):
    def __init__(self, ratings_file):
        self.ratings_df = pd.read_csv(ratings_file)
        self.users = self.ratings_df['user_id'].unique()
        self.books = self.ratings_df['book_id'].unique()
        
        # Create mapping dictionaries for user and book IDs
        self.user2idx = {user: idx for idx, user in enumerate(self.users)}
        self.book2idx = {book: idx for idx, book in enumerate(self.books)}
        
        self.user_ids = torch.tensor([self.user2idx[user] for user in self.ratings_df['user_id']])
        self.book_ids = torch.tensor([self.book2idx[book] for book in self.ratings_df['book_id']])
        self.ratings = torch.tensor(self.ratings_df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.book_ids[idx], self.ratings[idx]

class BookRecommender(nn.Module):
    def __init__(self, n_users, n_books, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.book_factors = nn.Embedding(n_books, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.book_biases = nn.Embedding(n_books, 1)
        
        # Initialize weights
        self.user_factors.weight.data.normal_(0, 0.1)
        self.book_factors.weight.data.normal_(0, 0.1)
        self.user_biases.weight.data.zero_()
        self.book_biases.weight.data.zero_()

    def forward(self, user_ids, book_ids):
        user_embeds = self.user_factors(user_ids)
        book_embeds = self.book_factors(book_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        book_bias = self.book_biases(book_ids).squeeze()
        
        # Compute dot product of user and book embeddings
        dot_products = (user_embeds * book_embeds).sum(dim=1)
        return dot_products + user_bias + book_bias + 3.0  # Add global mean rating

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create lists to store loss history
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids, book_ids, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_ids, book_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Store average training loss
        avg_train_loss = total_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, book_ids, ratings in val_loader:
                predictions = model(user_ids, book_ids)
                val_loss += criterion(predictions, ratings).item()
        
        # Store average validation loss
        avg_val_loss = val_loss/len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print('-' * 30)

    # Plot training results with actual history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load and prepare data
    dataset = BookRecommenderDataset('ratings.csv')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    
    # Initialize model
    model = BookRecommender(len(dataset.users), len(dataset.books))
    
    # Train the model
    train_model(model, train_loader, val_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'book_recommender.pth')
    
    # Save the dataset mappings
    torch.save({
        'user2idx': dataset.user2idx,
        'book2idx': dataset.book2idx,
        'n_users': len(dataset.users),
        'n_books': len(dataset.books)
    }, 'mappings.pth')

if __name__ == '__main__':
    main()
