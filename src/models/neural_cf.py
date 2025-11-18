"""
Neural Collaborative Filtering Implementation
Deep learning approach for recommendation systems
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering recommendation model"""
    
    def __init__(self, n_factors: int = 50, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.2, learning_rate: float = 0.001, 
                 random_state: int = 42):
        """
        Initialize Neural Collaborative Filtering model
        
        Args:
            n_factors: Number of embedding factors
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random state for reproducibility
        """
        self.n_factors = n_factors
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.n_users = 0
        self.n_items = 0
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.global_mean = 0
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self):
        """Build the neural collaborative filtering model"""
        # Input layers
        user_input = layers.Input(shape=(), name='user_input')
        item_input = layers.Input(shape=(), name='item_input')
        
        # Embedding layers
        user_embedding = layers.Embedding(
            self.n_users, 
            self.n_factors, 
            name='user_embedding'
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        item_embedding = layers.Embedding(
            self.n_items, 
            self.n_factors, 
            name='item_embedding'
        )(item_input)
        item_embedding = layers.Flatten()(item_embedding)
        
        # Concatenate embeddings
        concat = layers.Concatenate()([user_embedding, item_embedding])
        
        # Hidden layers
        x = concat
        for layer_size in self.hidden_layers:
            x = layers.Dense(layer_size, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='rating_output')(x)
        
        # Create model
        self.model = keras.Model(
            inputs=[user_input, item_input], 
            outputs=output,
            name='NeuralCollaborativeFiltering'
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, train_data: pd.DataFrame, user_to_idx: Dict, item_to_idx: Dict, 
            validation_data: pd.DataFrame = None, epochs: int = 50, batch_size: int = 256):
        """
        Fit the neural collaborative filtering model
        
        Args:
            train_data: Training data with columns ['user_idx', 'item_idx', 'rating']
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
            validation_data: Validation data for early stopping
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {v: k for k, v in user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}
        
        self.n_users = len(user_to_idx)
        self.n_items = len(item_to_idx)
        
        # Calculate global mean
        self.global_mean = train_data['rating'].mean()
        
        # Build model
        self._build_model()
        
        # Prepare training data
        X_train = [train_data['user_idx'].values, train_data['item_idx'].values]
        y_train = train_data['rating'].values
        
        # Prepare validation data if provided
        validation_split = None
        if validation_data is not None:
            X_val = [validation_data['user_idx'].values, validation_data['item_idx'].values]
            y_val = validation_data['rating'].values
            validation_split = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_split else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_split else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"Training Neural CF model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Predicted rating
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Make prediction
        prediction = self.model.predict([np.array([user_idx]), np.array([item_idx])], verbose=0)
        
        # Clamp to rating range
        prediction = max(1, min(5, prediction[0][0]))
        
        return prediction
    
    def predict_batch(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict ratings for multiple user-item pairs
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            
        Returns:
            Array of predicted ratings
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict([user_indices, item_indices], verbose=0)
        
        # Clamp to rating range
        predictions = np.clip(predictions.flatten(), 1, 5)
        
        return predictions
    
    def recommend(self, user_idx: int, n_recommendations: int = 10, 
                  exclude_rated: bool = True, rated_items: set = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            rated_items: Set of item indices already rated by user
            
        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get all items or unrated items
        if exclude_rated and rated_items is not None:
            candidate_items = list(set(range(self.n_items)) - rated_items)
        else:
            candidate_items = list(range(self.n_items))
        
        if len(candidate_items) == 0:
            return []
        
        # Create arrays for batch prediction
        user_array = np.full(len(candidate_items), user_idx)
        item_array = np.array(candidate_items)
        
        # Get predictions
        predictions = self.predict_batch(user_array, item_array)
        
        # Create list of (item_idx, prediction) tuples
        recommendations = list(zip(candidate_items, predictions))
        
        # Sort by predicted rating and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_user_embeddings(self, user_indices: np.ndarray = None) -> np.ndarray:
        """
        Get user embeddings from the trained model
        
        Args:
            user_indices: Specific user indices, if None returns all users
            
        Returns:
            User embeddings matrix
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting embeddings")
        
        if user_indices is None:
            user_indices = np.arange(self.n_users)
        
        # Get user embedding layer
        user_embedding_layer = self.model.get_layer('user_embedding')
        user_embeddings = user_embedding_layer(user_indices)
        
        return user_embeddings.numpy()
    
    def get_item_embeddings(self, item_indices: np.ndarray = None) -> np.ndarray:
        """
        Get item embeddings from the trained model
        
        Args:
            item_indices: Specific item indices, if None returns all items
            
        Returns:
            Item embeddings matrix
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting embeddings")
        
        if item_indices is None:
            item_indices = np.arange(self.n_items)
        
        # Get item embedding layer
        item_embedding_layer = self.model.get_layer('item_embedding')
        item_embeddings = item_embedding_layer(item_indices)
        
        return item_embeddings.numpy()
    
    def get_similar_items(self, item_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to a given item using embeddings
        
        Args:
            item_idx: Item index
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_idx, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model must be fitted before finding similar items")
        
        # Get item embeddings
        item_embeddings = self.get_item_embeddings()
        
        # Calculate cosine similarity
        target_embedding = item_embeddings[item_idx]
        similarities = np.dot(item_embeddings, target_embedding)
        
        # Create list of (item_idx, similarity) tuples
        similar_items = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity and return top n (excluding the item itself)
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items[1:n_similar+1]  # Exclude the item itself
    
    def get_similar_users(self, user_idx: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find users similar to a given user using embeddings
        
        Args:
            user_idx: User index
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_idx, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model must be fitted before finding similar users")
        
        # Get user embeddings
        user_embeddings = self.get_user_embeddings()
        
        # Calculate cosine similarity
        target_embedding = user_embeddings[user_idx]
        similarities = np.dot(user_embeddings, target_embedding)
        
        # Create list of (user_idx, similarity) tuples
        similar_users = [(i, similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity and return top n (excluding the user itself)
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[1:n_similar+1]  # Exclude the user itself
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        info = {
            'n_factors': self.n_factors,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'global_mean': self.global_mean,
            'total_params': self.model.count_params() if self.model is not None else 0
        }
        
        return info
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
