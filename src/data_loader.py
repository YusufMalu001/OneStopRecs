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
    
    # inside DataLoader class

SNAP_CATEGORY_KEYS = {
    "Electronics": "Electronics",
    "Home and Kitchen": "Home_and_Kitchen",
    "Beauty": "Beauty",
    "Sports and Outdoors": "Sports_and_Outdoors",
    "Toys and Games": "Toys_and_Games",
    "Automotive": "Automotive",
    "Musical Instruments": "Musical_Instruments",
}

def _safe_download(self, url: str, dest: str, timeout: int = 30):
    """Stream-download file robustly."""
    if os.path.exists(dest):
        return dest
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 32):
            if chunk:
                f.write(chunk)
    return dest

def load_amazon_category(self, snap_key: str, limit: int = None):
    """
    Download & load one SNAP category split. Returns (reviews_df, meta_df).
    snap_key: e.g. "Electronics" or "Home_and_Kitchen" (use SNAP_CATEGORY_KEYS to map).
    """
    base_reviews = f"https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{snap_key}_5.json.gz"
    base_meta = f"https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles/meta_{snap_key}.json.gz"

    reviews_path = os.path.join(self.data_dir, f"{snap_key}_reviews.json.gz")
    meta_path = os.path.join(self.data_dir, f"{snap_key}_meta.json.gz")

    # download safely
    self._safe_download(base_reviews, reviews_path)
    self._safe_download(base_meta, meta_path)

    # load reviews (JSON lines)
    reviews = pd.read_json(reviews_path, lines=True)
    # select and rename fields to match your pipeline
    keep_cols = [c for c in ["reviewerID", "asin", "overall", "unixReviewTime"] if c in reviews.columns]
    reviews = reviews[keep_cols].rename(columns={
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp"
    })
    if "timestamp" in reviews.columns:
        reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], unit="s", errors="coerce")

    if limit:
        reviews = reviews.head(limit)

    # load metadata
    meta = pd.read_json(meta_path, lines=True)
    # keep common fields if present
    meta_keep = [c for c in ["asin", "title", "brand", "categories"] if c in meta.columns]
    meta = meta[meta_keep].rename(columns={"asin": "item_id"})
    # tag top-level category (the label you want)
    meta["top_category"] = snap_key

    # ensure types are str for item ids
    meta["item_id"] = meta["item_id"].astype(str)
    reviews["item_id"] = reviews["item_id"].astype(str)
    reviews["user_id"] = reviews["user_id"].astype(str)

    return reviews, meta

def load_amazon_categories(self, categories: list = None, per_cat_limit: int = None):
    """
    Load multiple SNAP categories (your requested list). Returns concatenated (reviews, products).
    categories: friendly names like "Electronics", "Home and Kitchen", etc.
    per_cat_limit: optional limit applied per category for faster experiments.
    """
    if categories is None:
        categories = list(SNAP_CATEGORY_KEYS.keys())

    reviews_list = []
    meta_list = []
    for friendly in categories:
        if friendly not in SNAP_CATEGORY_KEYS:
            raise ValueError(f"Unknown category {friendly}. Valid keys: {list(SNAP_CATEGORY_KEYS.keys())}")
        snap_key = SNAP_CATEGORY_KEYS[friendly]
        print(f"Loading SNAP category: {friendly} -> {snap_key}")
        r, m = self.load_amazon_category(snap_key, limit=per_cat_limit)
        # attach a clean category label (friendly)
        m["category_label"] = friendly
        r["category_label"] = friendly
        reviews_list.append(r)
        meta_list.append(m)

    all_reviews = pd.concat(reviews_list, ignore_index=True)
    all_meta = pd.concat(meta_list, ignore_index=True).drop_duplicates(subset=["item_id"])
    print(f"Combined dataset: {len(all_reviews)} ratings, {len(all_meta)} products across {len(categories)} categories")
    return all_reviews, all_meta

    

def main():
    """Main function to demonstrate data loading"""
    loader = DataLoader()
    
    print("=== Loading MovieLens Dataset ===")
    ml_ratings, ml_movies, ml_tags = loader.load_movielens()
    
    ml_ratings = ml_ratings.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
    ml_movies = ml_movies.rename(columns={'movieId': 'item_id'})

    # Ensure types
    if 'user_id' in ml_ratings.columns:
        ml_ratings['user_id'] = ml_ratings['user_id'].astype(str)
    ml_ratings['item_id'] = ml_ratings['item_id'].astype(str)
    ml_movies['item_id'] = ml_movies['item_id'].astype(str)
    
    # Preprocess MovieLens data
    ml_data = loader.preprocess_data(ml_ratings, ml_movies)
    ml_stats = loader.get_dataset_stats(ml_data)
    print("MovieLens Stats:", ml_stats)
    
    print("\n=== Loading GoodBooks Dataset ===")
    gb_ratings, gb_books, gb_tags = loader.load_goodbooks()
    
    # Rename columns for consistency
    gb_ratings = gb_ratings.rename(columns={'book_id': 'item_id', 'user_id': 'user_id'})
    gb_books = gb_books.rename(columns={'book_id': 'item_id'})

    # Ensure types
    if 'user_id' in gb_ratings.columns:
        gb_ratings['user_id'] = gb_ratings['user_id'].astype(str)
    gb_ratings['item_id'] = gb_ratings['item_id'].astype(str)
    gb_books['item_id'] = gb_books['item_id'].astype(str)
    
    # Preprocess GoodBooks data
    gb_data = loader.preprocess_data(gb_ratings, gb_books)
    gb_stats = loader.get_dataset_stats(gb_data)
    print("GoodBooks Stats:", gb_stats)
    
    print("\n=== Loading Amazon SNAP Categories ===")
    chosen = [
        "Electronics",
        "Home and Kitchen",
        "Beauty",
        "Sports and Outdoors",
        "Toys and Games",
        "Automotive",
        "Musical Instruments",
    ]

    # per_cat_limit controls how many reviews to load per category (None => full)
    # set a limit for faster iteration if you like (e.g., 100000). Use None to load everything.
    amazon_ratings, amazon_products = loader.load_amazon_categories(categories=chosen, per_cat_limit=100000)

    # Ensure consistent column names/types for the pipeline
    # load_amazon_categories should already set 'user_id' and 'item_id', but normalize just in case
    amazon_ratings = amazon_ratings.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'})
    amazon_products = amazon_products.rename(columns={'asin': 'item_id'})

    amazon_ratings['user_id'] = amazon_ratings['user_id'].astype(str)
    amazon_ratings['item_id'] = amazon_ratings['item_id'].astype(str)
    amazon_products['item_id'] = amazon_products['item_id'].astype(str)

    # Convert rating to numeric and timestamp if present
    if 'rating' in amazon_ratings.columns:
        amazon_ratings['rating'] = pd.to_numeric(amazon_ratings['rating'], errors='coerce').fillna(0)
    if 'timestamp' in amazon_ratings.columns:
        amazon_ratings['timestamp'] = pd.to_datetime(amazon_ratings['timestamp'], errors='coerce')

    # Preprocess Amazon data
    amazon_data = loader.preprocess_data(amazon_ratings, amazon_products)
    amazon_stats = loader.get_dataset_stats(amazon_data)
    print("Amazon Stats:", amazon_stats)

if __name__ == "__main__":
    main()
