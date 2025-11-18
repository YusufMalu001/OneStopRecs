"""
Matrix Factorization Implementation
Includes SVD and NMF techniques for recommendation systems
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MatrixFactorization:
    """Matrix Factorization recommendation model"""
    
    def __init__(self, method: str = 'svd', n_factors: int = 50, random_state: int = 42):
        """
        Initialize Matrix Factorization model
        
        Args:
            method: 'svd' for SVD or 'nmf' for Non-negative Matrix Factorization
            n_factors: Number of latent factors
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.ratings_matrix = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.global_mean = 0
        
    def fit(self, train_data: pd.DataFrame, user_to_idx: Dict, item_to_idx: Dict):
        """
        Fit the matrix factorization model
        
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
        
        # Calculate global mean
        self.global_mean = np.mean(self.ratings_matrix[self.ratings_matrix > 0])
        
        # Create mask for non-zero ratings
        mask = self.ratings_matrix > 0
        
        if self.method == 'svd':
            self._fit_svd(mask)
        elif self.method == 'nmf':
            self._fit_nmf(mask)
        else:
            raise ValueError("Method must be 'svd' or 'nmf'")
    
    def _fit_svd(self, mask: np.ndarray):
        """Fit SVD model"""
        print(f"Fitting SVD with {self.n_factors} factors...")
        
        # Create a copy of ratings matrix for SVD
        ratings_for_svd = self.ratings_matrix.copy()
        
        # Fill missing values with global mean for SVD
        ratings_for_svd[~mask] = self.global_mean
        
        # Apply SVD
        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        
        # Fit and transform
        self.user_factors = self.model.fit_transform(ratings_for_svd)
        self.item_factors = self.model.components_.T
        
        print(f"SVD explained variance ratio: {self.model.explained_variance_ratio_.sum():.4f}")
    
    def _fit_nmf(self, mask: np.ndarray):
        """Fit NMF model"""
        print(f"Fitting NMF with {self.n_factors} factors...")
        
        # Create a copy of ratings matrix for NMF
        ratings_for_nmf = self.ratings_matrix.copy()
        
        # Fill missing values with small positive value for NMF
        ratings_for_nmf[~mask] = 0.1
        
        # Apply NMF
        self.model = NMF(
            n_components=self.n_factors,
            random_state=self.random_state,
            max_iter=200
        )
        
        # Fit and transform
        self.user_factors = self.model.fit_transform(ratings_for_nmf)
        self.item_factors = self.model.components_.T
        
        print(f"NMF reconstruction error: {self.model.reconstruction_err_:.4f}")
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Predicted rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Calculate dot product of user and item factors
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Clamp to rating range
        prediction = max(1, min(5, prediction))
        
        return prediction
    
    def recommend(self, user_idx: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        user_ratings = self.ratings_matrix[user_idx]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Calculate predictions for all unrated items
        user_factor = self.user_factors[user_idx]
        item_factors_unrated = self.item_factors[unrated_items]
        
        predictions = np.dot(item_factors_unrated, user_factor)
        
        # Create list of (item_idx, prediction) tuples
        recommendations = list(zip(unrated_items, predictions))
        
        # Sort by predicted rating and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_similar_items(self, item_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to a given item
        
        Args:
            item_idx: Item index
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_idx, similarity_score) tuples
        """
        if self.item_factors is None:
            raise ValueError("Model must be fitted before finding similar items")
        
        # Calculate cosine similarity between target item and all other items
        target_factor = self.item_factors[item_idx]
        similarities = np.dot(self.item_factors, target_factor)
        
        # Create list of (item_idx, similarity) tuples
        similar_items = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity and return top n (excluding the item itself)
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items[1:n_similar+1]  # Exclude the item itself
    
    def get_similar_users(self, user_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find users similar to a given user
        
        Args:
            user_idx: User index
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_idx, similarity_score) tuples
        """
        if self.user_factors is None:
            raise ValueError("Model must be fitted before finding similar users")
        
        # Calculate cosine similarity between target user and all other users
        target_factor = self.user_factors[user_idx]
        similarities = np.dot(self.user_factors, target_factor)
        
        # Create list of (user_idx, similarity) tuples
        similar_users = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity and return top n (excluding the user itself)
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[1:n_similar+1]  # Exclude the user itself
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        info = {
            'method': self.method,
            'n_factors': self.n_factors,
            'n_users': self.ratings_matrix.shape[0] if self.ratings_matrix is not None else 0,
            'n_items': self.ratings_matrix.shape[1] if self.ratings_matrix is not None else 0,
            'sparsity': 1 - np.count_nonzero(self.ratings_matrix) / self.ratings_matrix.size if self.ratings_matrix is not None else 0,
            'global_mean': self.global_mean
        }
        
        if self.model is not None:
            if self.method == 'svd':
                info['explained_variance_ratio'] = self.model.explained_variance_ratio_.sum()
            elif self.method == 'nmf':
                info['reconstruction_error'] = self.model.reconstruction_err_
        
        return info

class SVDModel(MatrixFactorization):
    """SVD-based Matrix Factorization"""
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        super().__init__(method='svd', n_factors=n_factors, random_state=random_state)

class NMFModel(MatrixFactorization):
    """Non-negative Matrix Factorization"""
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        super().__init__(method='nmf', n_factors=n_factors, random_state=random_state)
