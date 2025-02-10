import pandas as pd
import torch
import random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from rec_algo import BookRecommender

class BookRecommendationInterface:
    def __init__(self):
        try:
            # Load necessary data
            self.books_df = pd.read_csv('books.csv')
            self.book_tags_df = pd.read_csv('book_tags.csv')
            self.tags_df = pd.read_csv('tags.csv')  # Load the tags mapping file
            self.ratings_df = pd.read_csv('ratings.csv')
            
            # Load the mappings used during training
            mappings = torch.load('mappings.pth')
            self.book2idx = mappings['book2idx']
            self.n_users = mappings['n_users']
            self.n_books = mappings['n_books']
            
            # Calculate book popularity
            self.book_popularity = self.ratings_df.groupby('book_id').agg({
                'rating': ['count', 'mean']
            }).sort_values(('rating', 'count'), ascending=False)
            
            # Load the trained model
            self.load_model()
            
            # Create genre to book mapping
            self.genre_to_books = self.create_genre_mapping()
            
            # Extract item embeddings from the model
            self.item_embeddings = self.model.book_factors.weight.detach().numpy()
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            print("Please check if all required files exist and have the correct format:")
            print("- books.csv")
            print("- book_tags.csv")
            print("- tags.csv")
            print("- ratings.csv")
            print("- book_recommender.pth")
            print("- mappings.pth")
            raise
    
    def load_model(self):
        # Load the saved model with correct dimensions from mappings
        self.model = BookRecommender(self.n_users, self.n_books)
        self.model.load_state_dict(torch.load('book_recommender.pth', weights_only=True))
        self.model.eval()
    
    def create_genre_mapping(self):
        """Create a dictionary mapping genres to book IDs"""
        # Define main genres we want to include
        main_genres = {
            'fiction', 'fantasy', 'science-fiction', 'mystery', 'thriller', 
            'romance', 'historical-fiction', 'non-fiction', 'biography',
            'history', 'science', 'philosophy', 'poetry', 'drama',
            'horror', 'adventure', 'classics', 'contemporary'
        }
        
        # Merge book_tags with tags to get the actual tag names
        merged_tags = pd.merge(
            self.book_tags_df,
            self.tags_df,
            left_on='tag_id',
            right_on='tag_id'
        )
        
        # Filter for main genres and create mapping
        genre_books = {}
        for _, row in merged_tags.iterrows():
            genre = row['tag_name'].lower().strip()
            if genre in main_genres:
                if genre not in genre_books:
                    genre_books[genre] = []
                genre_books[genre].append(row['goodreads_book_id'])
        
        # Only keep genres that have enough books (at least 20)
        genre_books = {genre: books for genre, books in genre_books.items() 
                      if len(books) >= 20}
        
        return genre_books
    
    def get_available_genres(self):
        return sorted(list(self.genre_to_books.keys()))
    
    def get_popular_books_by_genre(self, genre, n=10, exclude_ids=None):
        """Get n most popular books from the selected genre"""
        if genre not in self.genre_to_books:
            return []
            
        if exclude_ids is None:
            exclude_ids = set()
        
        # Get all books in the genre
        genre_book_ids = set(self.genre_to_books[genre])
        
        # Filter popularity data for this genre's books
        genre_popularity = self.book_popularity[
            self.book_popularity.index.isin(genre_book_ids - exclude_ids)
        ]
        
        # Get top N popular books
        popular_books = []
        for book_id in genre_popularity.index[:min(n * 2, len(genre_popularity))]:
            if len(popular_books) >= n:
                break
            if book_id in exclude_ids:
                continue
                
            try:
                book_info = self.books_df[self.books_df['book_id'] == book_id].iloc[0]
                popular_books.append({
                    'id': book_id,
                    'title': book_info['title'],
                    'author': book_info['authors'],
                    'rating_count': genre_popularity.loc[book_id, ('rating', 'count')],
                    'avg_rating': genre_popularity.loc[book_id, ('rating', 'mean')]
                })
            except IndexError:
                continue
                
        return popular_books[:n]
    
    def get_recommendations(self, user_ratings, genre, n=5):
        """Get book recommendations based on user ratings"""
        # Convert book IDs to indices using the mapping
        try:
            book_indices = [self.book2idx[rating['book_id']] for rating in user_ratings]
            book_ids = torch.tensor(book_indices)
            ratings = torch.tensor([rating['rating'] for rating in user_ratings])
            
            # Create a temporary user embedding based on these ratings
            with torch.no_grad():
                user_embed = self.model.user_factors.weight.mean(0)  # Start with average user
                for book_idx, rating in zip(book_indices, ratings):
                    book_embed = self.model.book_factors(torch.tensor([book_idx]))
                    user_embed += (rating - 3.0) * book_embed[0] * 0.1
            
            # Get predictions for all books in the genre
            genre_books = self.genre_to_books[genre]
            predictions = []
            rated_books = {r['book_id'] for r in user_ratings}
            
            for book_id in genre_books:
                if book_id not in rated_books and book_id in self.book2idx:  # Check if book is in mapping
                    book_idx = self.book2idx[book_id]
                    with torch.no_grad():
                        pred = self.model.forward(
                            torch.tensor([0]),  # Dummy user ID
                            torch.tensor([book_idx])
                        )
                    predictions.append((book_id, pred.item()))
            
            # Sort and get top N recommendations
            predictions.sort(key=lambda x: x[1], reverse=True)
            recommendations = []
            for book_id, score in predictions[:n]:
                book_info = self.books_df[self.books_df['book_id'] == book_id].iloc[0]
                recommendations.append({
                    'title': book_info['title'],
                    'author': book_info['authors'],
                    'score': score
                })
            return recommendations
        except KeyError as e:
            print(f"Warning: Some books were not found in the training data: {e}")
            return []

    def get_book_ratings_count(self, title):
        """Get the number of ratings for a specific book title"""
        try:
            # Find the book ID for this title
            book_info = self.books_df[self.books_df['title'].str.contains(title, case=False, na=False)]
            if len(book_info) == 0:
                print(f"No book found with title containing '{title}'")
                return
            
            if len(book_info) > 1:
                print("Multiple books found:")
                for _, book in book_info.iterrows():
                    print(f"- {book['title']} by {book['authors']}")
                return
            
            book_id = book_info.iloc[0]['book_id']
            
            # Get ratings count from popularity DataFrame
            if book_id in self.book_popularity.index:
                count = self.book_popularity.loc[book_id, ('rating', 'count')]
                avg_rating = self.book_popularity.loc[book_id, ('rating', 'mean')]
                print(f"\nBook: {book_info.iloc[0]['title']}")
                print(f"Author: {book_info.iloc[0]['authors']}")
                print(f"Number of ratings: {int(count)}")
                print(f"Average rating: {avg_rating:.2f}")
            else:
                print("No ratings found for this book")
            
        except Exception as e:
            print(f"Error: {e}")

    def get_recommendations_ridge(self, user_ratings, genre, n=5):
        """Get book recommendations using Ridge regression"""
        try:
            # Prepare data for ridge regression
            rated_book_indices = [self.book2idx[rating['book_id']] for rating in user_ratings]
            X = self.item_embeddings[rated_book_indices]
            y = np.array([rating['rating'] for rating in user_ratings])

            # Fit ridge regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)

            # Get user preferences from coefficients
            user_preferences = ridge_model.coef_

            # Get predictions for all books in the genre
            genre_books = self.genre_to_books[genre]
            predictions = []
            rated_books = {r['book_id'] for r in user_ratings}

            for book_id in genre_books:
                if book_id not in rated_books and book_id in self.book2idx:
                    book_idx = self.book2idx[book_id]
                    pred = np.dot(self.item_embeddings[book_idx], user_preferences)
                    predictions.append((book_id, float(pred)))

            # Sort and get top N recommendations
            predictions.sort(key=lambda x: x[1], reverse=True)
            recommendations = []
            for book_id, score in predictions[:n]:
                book_info = self.books_df[self.books_df['book_id'] == book_id].iloc[0]
                recommendations.append({
                    'title': book_info['title'],
                    'author': book_info['authors'],
                    'score': score
                })
            return recommendations
        except Exception as e:
            print(f"Error in ridge recommendations: {e}")
            return []

    def get_similar_books(self, book_title, n=10):
        """Find similar books using cosine similarity"""
        try:
            # Find the book ID for this title
            book_info = self.books_df[self.books_df['title'].str.contains(book_title, case=False, na=False)]
            if len(book_info) == 0:
                return []

            book_id = book_info.iloc[0]['book_id']
            if book_id not in self.book2idx:
                return []

            # Get book vector and calculate similarities
            book_idx = self.book2idx[book_id]
            book_vector = self.item_embeddings[book_idx].reshape(1, -1)
            similarities = cosine_similarity(self.item_embeddings, book_vector).flatten()

            # Get top similar books
            similar_indices = np.argsort(-similarities)[1:n+1]
            recommendations = []

            for idx in similar_indices:
                # Convert model index back to book_id
                for book_id, model_idx in self.book2idx.items():
                    if model_idx == idx:
                        book_info = self.books_df[self.books_df['book_id'] == book_id].iloc[0]
                        recommendations.append({
                            'title': book_info['title'],
                            'author': book_info['authors'],
                            'similarity': float(similarities[idx])
                        })
                        break

            return recommendations
        except Exception as e:
            print(f"Error finding similar books: {e}")
            return []

def main():
    interface = BookRecommendationInterface()
    
    # Show available genres
    print("Available genres:")
    genres = interface.get_available_genres()
    for i, genre in enumerate(genres, 1):
        print(f"{i}. {genre}")
    
    # Get genre selection
    while True:
        try:
            genre_idx = int(input("\nSelect a genre (enter number): ")) - 1
            if 0 <= genre_idx < len(genres):
                break
            print("Please enter a valid number.")
        except ValueError:
            print("Please enter a valid number.")
    
    selected_genre = genres[genre_idx]
    user_ratings = []
    
    # First round - Popular books
    print(f"\nFirst, please rate these popular {selected_genre} books (1-5 stars):")
    popular_books = interface.get_popular_books_by_genre(selected_genre)
    
    for book in popular_books:
        print(f"\nTitle: {book['title']}")
        print(f"Author: {book['author']}")
        print(f"Average Rating: {book['avg_rating']:.2f} from {book['rating_count']} ratings")
        
        while True:
            try:
                rating = float(input("Your rating (1-5): "))
                if 1 <= rating <= 5:
                    break
                print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
                
        user_ratings.append({
            'book_id': book['id'],
            'rating': rating
        })
    
    # Second round - More books based on initial ratings
    print(f"\nBased on your ratings, please rate these additional {selected_genre} books:")
    rated_book_ids = {rating['book_id'] for rating in user_ratings}
    more_books = interface.get_popular_books_by_genre(selected_genre, exclude_ids=rated_book_ids)
    
    for book in more_books:
        print(f"\nTitle: {book['title']}")
        print(f"Author: {book['author']}")
        print(f"Average Rating: {book['avg_rating']:.2f} from {book['rating_count']} ratings")
        
        while True:
            try:
                rating = float(input("Your rating (1-5): "))
                if 1 <= rating <= 5:
                    break
                print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
                
        user_ratings.append({
            'book_id': book['id'],
            'rating': rating
        })
    
    # Get final recommendations using Ridge regression
    print("\nBased on all your ratings, here are some recommendations (Ridge regression):")
    recommendations = interface.get_recommendations_ridge(user_ratings, selected_genre)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   by {rec['author']}")
        print(f"   Predicted Rating: {rec['score']:.2f}")

    # Example of finding similar books
    print("\nFinding similar books to 'American Gods':")
    similar_books = interface.get_similar_books("American Gods")
    for i, book in enumerate(similar_books, 1):
        print(f"\n{i}. {book['title']}")
        print(f"   by {book['author']}")
        print(f"   Similarity: {book['similarity']:.2f}")

    # Add this to the main() function or run separately:
    interface.get_book_ratings_count("American Gods")

if __name__ == "__main__":
    main() 