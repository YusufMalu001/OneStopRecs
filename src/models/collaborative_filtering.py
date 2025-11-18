"""
Collaborative Filtering Implementation
Includes User-based and Item-based Collaborative Filtering
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """Collaborative Filtering recommendation model"""
    
    def __init__(self, method: str = 'user', similarity_metric: str = 'cosine'):
        """
        Initialize Collaborative Filtering model
        
        Args:
            method: 'user' for user-based or 'item' for item-based CF
            similarity_metric: 'cosine' or 'pearson'
        """
        self.method = method
        self.similarity_metric = similarity_metric
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.ratings_matrix = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        
    def fit(self, train_data: pd.DataFrame, user_to_idx: Dict, item_to_idx: Dict):
        """
        Fit the collaborative filtering model
        
        Args:
            train_data: Training data with columns ['user_idx', 'item_idx', 'rating']
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
        """
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {v: k for k, v in user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}
        
        n_users = len(user_to_idx)
        n_items = len(item_to_idx)
        
        # Create ratings matrix
        self.ratings_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            rating = row['rating']
            self.ratings_matrix[user_idx, item_idx] = rating
        
        # Calculate similarity matrices
        if self.method == 'user':
            self._calculate_user_similarity()
        else:
            self._calculate_item_similarity()
    
    def _calculate_user_similarity(self):
        """Calculate user-user similarity matrix"""
        print("Calculating user similarity matrix...")
        
        if self.similarity_metric == 'cosine':
            # Handle zero vectors by adding small epsilon
            ratings_with_epsilon = self.ratings_matrix + 1e-8
            self.user_similarity_matrix = cosine_similarity(ratings_with_epsilon)
        elif self.similarity_metric == 'pearson':
            self.user_similarity_matrix = np.corrcoef(self.ratings_matrix)
            # Replace NaN values with 0
            self.user_similarity_matrix = np.nan_to_num(self.user_similarity_matrix)
        
        # Set diagonal to 0 (users are not similar to themselves)
        np.fill_diagonal(self.user_similarity_matrix, 0)
    
    def _calculate_item_similarity(self):
        """Calculate item-item similarity matrix"""
        print("Calculating item similarity matrix...")
        
        if self.similarity_metric == 'cosine':
            # Handle zero vectors by adding small epsilon
            ratings_with_epsilon = self.ratings_matrix.T + 1e-8
            self.item_similarity_matrix = cosine_similarity(ratings_with_epsilon)
        elif self.similarity_metric == 'pearson':
            self.item_similarity_matrix = np.corrcoef(self.ratings_matrix.T)
            # Replace NaN values with 0
            self.item_similarity_matrix = np.nan_to_num(self.item_similarity_matrix)
        
        # Set diagonal to 0 (items are not similar to themselves)
        np.fill_diagonal(self.item_similarity_matrix, 0)
    
    def predict(self, user_idx: int, item_idx: int, k: int = 50) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_idx: User index
            item_idx: Item index
            k: Number of similar users/items to consider
            
        Returns:
            Predicted rating
        """
        if self.method == 'user':
            return self._predict_user_based(user_idx, item_idx, k)
        else:
            return self._predict_item_based(user_idx, item_idx, k)
    
    def _predict_user_based(self, user_idx: int, item_idx: int, k: int) -> float:
        """User-based collaborative filtering prediction"""
        # Get user's ratings
        user_ratings = self.ratings_matrix[user_idx]
        
        # If user hasn't rated the item, we can't predict
        if user_ratings[item_idx] != 0:
            return user_ratings[item_idx]
        
        # Find similar users who have rated this item
        similar_users = self.user_similarity_matrix[user_idx]
        users_who_rated = np.where(self.ratings_matrix[:, item_idx] != 0)[0]
        
        if len(users_who_rated) == 0:
            return self.ratings_matrix.mean()  # Return global average
        
        # Get similarities for users who rated the item
        similarities = similar_users[users_who_rated]
        ratings = self.ratings_matrix[users_who_rated, item_idx]
        
        # Sort by similarity and take top k
        sorted_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[sorted_indices]
        top_ratings = ratings[sorted_indices]
        
        # Filter out zero similarities
        valid_indices = top_similarities > 0
        if not np.any(valid_indices):
            return self.ratings_matrix.mean()
        
        top_similarities = top_similarities[valid_indices]
        top_ratings = top_ratings[valid_indices]
        
        # Weighted average prediction
        if np.sum(np.abs(top_similarities)) == 0:
            return self.ratings_matrix.mean()
        
        prediction = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        return max(1, min(5, prediction))  # Clamp to rating range
    
    def _predict_item_based(self, user_idx: int, item_idx: int, k: int) -> float:
        """Item-based collaborative filtering prediction"""
        # Get user's ratings
        user_ratings = self.ratings_matrix[user_idx]
        
        # If user has already rated the item, return that rating
        if user_ratings[item_idx] != 0:
            return user_ratings[item_idx]
        
        # Find items the user has rated
        rated_items = np.where(user_ratings != 0)[0]
        
        if len(rated_items) == 0:
            return self.ratings_matrix.mean()  # Return global average
        
        # Get similarities between target item and rated items
        similarities = self.item_similarity_matrix[item_idx, rated_items]
        ratings = user_ratings[rated_items]
        
        # Sort by similarity and take top k
        sorted_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[sorted_indices]
        top_ratings = ratings[sorted_indices]
        
        # Filter out zero similarities
        valid_indices = top_similarities > 0
        if not np.any(valid_indices):
            return self.ratings_matrix.mean()
        
        top_similarities = top_similarities[valid_indices]
        top_ratings = top_ratings[valid_indices]
        
        # Weighted average prediction
        if np.sum(np.abs(top_similarities)) == 0:
            return self.ratings_matrix.mean()
        
        prediction = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        return max(1, min(5, prediction))  # Clamp to rating range
    
    def recommend(self, user_idx: int, n_recommendations: int = 10, k: int = 50) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations to generate
            k: Number of similar users/items to consider
            
        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        user_ratings = self.ratings_matrix[user_idx]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        predictions = []
        for item_idx in unrated_items:
            pred_rating = self.predict(user_idx, item_idx, k)
            predictions.append((item_idx, pred_rating))
        
        # Sort by predicted rating and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            'method': self.method,
            'similarity_metric': self.similarity_metric,
            'n_users': self.ratings_matrix.shape[0] if self.ratings_matrix is not None else 0,
            'n_items': self.ratings_matrix.shape[1] if self.ratings_matrix is not None else 0,
            'sparsity': 1 - np.count_nonzero(self.ratings_matrix) / self.ratings_matrix.size if self.ratings_matrix is not None else 0
        }

class UserBasedCF(CollaborativeFiltering):
    """User-based Collaborative Filtering"""
    def __init__(self, similarity_metric: str = 'cosine'):
        super().__init__(method='user', similarity_metric=similarity_metric)

class ItemBasedCF(CollaborativeFiltering):
    """Item-based Collaborative Filtering"""
    def __init__(self, similarity_metric: str = 'cosine'):
        super().__init__(method='item', similarity_metric=similarity_metric)
