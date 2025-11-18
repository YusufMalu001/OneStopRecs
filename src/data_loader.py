"""
Data Loading and Preprocessing Module for Recommender System Project
Handles loading and preprocessing of MovieLens, Amazon Reviews, and GoodBooks datasets
"""

import pandas as pd
import numpy as np
import os
import requests
import zipfile
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class to handle loading and preprocessing of recommendation datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_dir()
        
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download_movielens(self) -> str:
        """Download MovieLens 100K dataset"""
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        filename = os.path.join(self.data_dir, "ml-latest-small.zip")
        
        if not os.path.exists(filename):
            print("Downloading MovieLens dataset...")
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("MovieLens dataset downloaded and extracted successfully!")
        
        return os.path.join(self.data_dir, "ml-latest-small")
    
    def load_movielens(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load MovieLens dataset"""
        dataset_path = self.download_movielens()

        # Load ratings
        ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

        # Take subset for faster processing (first 50k ratings)
        ratings = ratings.head(50000)

        # Load movies
        movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

        # Load tags (if available)
        tags_path = os.path.join(dataset_path, "tags.csv")
        if os.path.exists(tags_path):
            tags = pd.read_csv(tags_path)
        else:
            tags = pd.DataFrame()

        print(f"MovieLens dataset loaded (subset): {len(ratings)} ratings, {len(movies)} movies")
        return ratings, movies, tags
    
    def download_goodbooks(self) -> str:
        """Download GoodBooks-10K dataset"""
        url = "https://github.com/zygmuntz/goodbooks-10k/archive/master.zip"
        filename = os.path.join(self.data_dir, "goodbooks-10k.zip")
        
        if not os.path.exists(filename):
            print("Downloading GoodBooks-10K dataset...")
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("GoodBooks-10K dataset downloaded and extracted successfully!")
        
        return os.path.join(self.data_dir, "goodbooks-10k-master")
    
    def load_goodbooks(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load GoodBooks-10K dataset"""
        dataset_path = self.download_goodbooks()

        # Load ratings
        ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

        # Take subset for faster processing (first 30k ratings)
        ratings = ratings.head(30000)

        # Load books
        books = pd.read_csv(os.path.join(dataset_path, "books.csv"))

        # Load book tags
        book_tags_path = os.path.join(dataset_path, "book_tags.csv")
        if os.path.exists(book_tags_path):
            book_tags = pd.read_csv(book_tags_path)
        else:
            book_tags = pd.DataFrame()

        print(f"GoodBooks dataset loaded (subset): {len(ratings)} ratings, {len(books)} books")
        return ratings, books, book_tags
    
    def create_sample_amazon_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample Amazon-style data for demonstration"""
        print("Creating sample Amazon Reviews dataset...")

        # Generate synthetic Amazon-style data (reduced size for speed)
        np.random.seed(42)
        n_users = 5000
        n_products = 2000
        n_ratings = 20000
        
        # Generate user and product IDs
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        product_ids = np.random.randint(1, n_products + 1, n_ratings)
        
        # Generate ratings (1-5 scale)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        
        # Generate timestamps
        timestamps = pd.date_range('2018-01-01', '2018-12-31', periods=n_ratings)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'product_id': product_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Create products dataframe
        products_df = pd.DataFrame({
            'product_id': range(1, n_products + 1),
            'title': [f'Product {i}' for i in range(1, n_products + 1)],
            'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports'], n_products)
        })
        
        print(f"Sample Amazon dataset created: {len(ratings_df)} ratings, {len(products_df)} products")
        return ratings_df, products_df
    
    def preprocess_data(self, ratings: pd.DataFrame, items: pd.DataFrame = None) -> Dict[str, Any]:
        """Preprocess the data for recommendation models"""
        
        # Remove duplicate ratings (keep the latest)
        if 'timestamp' in ratings.columns:
            ratings = ratings.sort_values('timestamp').drop_duplicates(['user_id', 'item_id'], keep='last')
        
        # Filter users and items with minimum interactions
        min_user_ratings = 5
        min_item_ratings = 5
        
        # Count ratings per user and item
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        
        # Filter users and items
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        
        ratings_filtered = ratings[
            (ratings['user_id'].isin(valid_users)) & 
            (ratings['item_id'].isin(valid_items))
        ].copy()
        
        # Create user and item mappings
        unique_users = ratings_filtered['user_id'].unique()
        unique_items = ratings_filtered['item_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # Map to indices
        ratings_filtered['user_idx'] = ratings_filtered['user_id'].map(user_to_idx)
        ratings_filtered['item_idx'] = ratings_filtered['item_id'].map(item_to_idx)

        # Create reverse mappings
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}

        print(f"Data preprocessed: {len(ratings_filtered)} ratings, {len(unique_users)} users, {len(unique_items)} items")

        return {
            'ratings': ratings_filtered,
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item,
            'n_users': len(unique_users),
            'n_items': len(unique_items),
            'items': items
        }
    
    def train_test_split(self, ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        # Sort by timestamp if available, otherwise random
        if 'timestamp' in ratings.columns:
            ratings_sorted = ratings.sort_values('timestamp')
        else:
            ratings_sorted = ratings.sample(frac=1, random_state=random_state)
        
        train_data, test_data = train_test_split(
            ratings_sorted, 
            test_size=test_size, 
            random_state=random_state,
            stratify=ratings_sorted['rating'] if len(ratings_sorted['rating'].unique()) > 1 else None
        )
        
        print(f"Data split: {len(train_data)} train samples, {len(test_data)} test samples")
        return train_data, test_data
    
    def get_dataset_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        ratings = data['ratings']
        
        stats = {
            'n_ratings': len(ratings),
            'n_users': data['n_users'],
            'n_items': data['n_items'],
            'sparsity': 1 - (len(ratings) / (data['n_users'] * data['n_items'])),
            'avg_rating': ratings['rating'].mean(),
            'rating_distribution': ratings['rating'].value_counts().to_dict(),
            'ratings_per_user': ratings.groupby('user_id').size().describe().to_dict(),
            'ratings_per_item': ratings.groupby('item_id').size().describe().to_dict()
        }
        
        return stats

def main():
    """Main function to demonstrate data loading"""
    loader = DataLoader()
    
    print("=== Loading MovieLens Dataset ===")
    ml_ratings, ml_movies, ml_tags = loader.load_movielens()
    
    # Rename columns for consistency
    ml_ratings = ml_ratings.rename(columns={'movieId': 'item_id'})
    ml_movies = ml_movies.rename(columns={'movieId': 'item_id'})
    
    # Preprocess MovieLens data
    ml_data = loader.preprocess_data(ml_ratings, ml_movies)
    ml_stats = loader.get_dataset_stats(ml_data)
    print("MovieLens Stats:", ml_stats)
    
    print("\n=== Loading GoodBooks Dataset ===")
    gb_ratings, gb_books, gb_tags = loader.load_goodbooks()
    
    # Rename columns for consistency
    gb_ratings = gb_ratings.rename(columns={'book_id': 'item_id'})
    gb_books = gb_books.rename(columns={'book_id': 'item_id'})
    
    # Preprocess GoodBooks data
    gb_data = loader.preprocess_data(gb_ratings, gb_books)
    gb_stats = loader.get_dataset_stats(gb_data)
    print("GoodBooks Stats:", gb_stats)
    
    print("\n=== Creating Sample Amazon Dataset ===")
    amazon_ratings, amazon_products = loader.create_sample_amazon_data()
    
    # Rename columns for consistency
    amazon_ratings = amazon_ratings.rename(columns={'product_id': 'item_id'})
    amazon_products = amazon_products.rename(columns={'product_id': 'item_id'})
    
    # Preprocess Amazon data
    amazon_data = loader.preprocess_data(amazon_ratings, amazon_products)
    amazon_stats = loader.get_dataset_stats(amazon_data)
    print("Amazon Stats:", amazon_stats)

if __name__ == "__main__":
    main()
