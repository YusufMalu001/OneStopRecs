"""
Evaluation Module for Recommender System Project
Implements various evaluation metrics for recommendation systems
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    """Comprehensive evaluation framework for recommendation systems"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_precision_at_k(self, y_true: List[set], y_pred: List[List], k: int = 10) -> float:
        """
        Calculate Precision@K
        
        Args:
            y_true: List of sets of relevant items for each user
            y_pred: List of predicted item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average precision@k across all users
        """
        precisions = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top k predictions
            top_k_pred = set(pred_items[:k])
            
            # Calculate precision
            if len(top_k_pred) > 0:
                precision = len(true_items.intersection(top_k_pred)) / len(top_k_pred)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def calculate_recall_at_k(self, y_true: List[set], y_pred: List[List], k: int = 10) -> float:
        """
        Calculate Recall@K
        
        Args:
            y_true: List of sets of relevant items for each user
            y_pred: List of predicted item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average recall@k across all users
        """
        recalls = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top k predictions
            top_k_pred = set(pred_items[:k])
            
            # Calculate recall
            recall = len(true_items.intersection(top_k_pred)) / len(true_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def calculate_f1_at_k(self, y_true: List[set], y_pred: List[List], k: int = 10) -> float:
        """
        Calculate F1-Score@K
        
        Args:
            y_true: List of sets of relevant items for each user
            y_pred: List of predicted item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average F1-score@k across all users
        """
        precision = self.calculate_precision_at_k(y_true, y_pred, k)
        recall = self.calculate_recall_at_k(y_true, y_pred, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ndcg_at_k(self, y_true: List[set], y_pred: List[List], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K
        
        Args:
            y_true: List of sets of relevant items for each user
            y_pred: List of predicted item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average NDCG@k across all users
        """
        ndcgs = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top k predictions
            top_k_pred = pred_items[:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Calculate IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(true_items), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def calculate_hit_rate_at_k(self, y_true: List[set], y_pred: List[List], k: int = 10) -> float:
        """
        Calculate Hit Rate@K (percentage of users with at least one relevant recommendation)
        
        Args:
            y_true: List of sets of relevant items for each user
            y_pred: List of predicted item lists for each user
            k: Number of top recommendations to consider
            
        Returns:
            Hit rate@k across all users
        """
        hits = 0
        total_users = 0
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            total_users += 1
            
            # Take top k predictions
            top_k_pred = set(pred_items[:k])
            
            # Check if there's at least one hit
            if len(true_items.intersection(top_k_pred)) > 0:
                hits += 1
        
        return hits / total_users if total_users > 0 else 0.0
    
    def evaluate_rating_prediction(self, model: Any, test_data: pd.DataFrame, 
                                 user_to_idx: Dict, item_to_idx: Dict) -> Dict[str, float]:
        """
        Evaluate model on rating prediction task
        
        Args:
            model: Trained recommendation model
            test_data: Test data with columns ['user_idx', 'item_idx', 'rating']
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating rating prediction...")
        
        y_true = test_data['rating'].values
        y_pred = []
        
        for _, row in test_data.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            pred_rating = model.predict(user_idx, item_idx)
            y_pred.append(pred_rating)
        
        y_pred = np.array(y_pred)
        
        metrics = {
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAE': self.calculate_mae(y_true, y_pred)
        }
        
        return metrics
    
    def evaluate_ranking(self, model: Any, test_data: pd.DataFrame, 
                        user_to_idx: Dict, item_to_idx: Dict, 
                        k_values: List[int] = [5, 10, 20], 
                        rating_threshold: float = 4.0) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on ranking task
        
        Args:
            model: Trained recommendation model
            test_data: Test data with columns ['user_idx', 'item_idx', 'rating']
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
            k_values: List of k values for evaluation
            rating_threshold: Threshold for considering an item as relevant
            
        Returns:
            Dictionary of evaluation metrics for each k value
        """
        print("Evaluating ranking performance...")
        
        # Group test data by user
        user_test_data = test_data.groupby('user_idx')
        
        y_true = []
        y_pred = []
        
        for user_idx, user_data in user_test_data:
            # Get relevant items (high ratings)
            relevant_items = set(user_data[user_data['rating'] >= rating_threshold]['item_idx'].tolist())
            
            if len(relevant_items) == 0:
                continue
            
            # Generate recommendations
            try:
                recommendations = model.recommend(user_idx, n_recommendations=max(k_values))
                pred_items = [item_idx for item_idx, _ in recommendations]
            except:
                # Fallback: predict ratings for all items and sort
                all_items = set(range(len(item_to_idx)))
                rated_items = set(user_data['item_idx'].tolist())
                unrated_items = all_items - rated_items
                
                predictions = []
                for item_idx in unrated_items:
                    try:
                        pred_rating = model.predict(user_idx, item_idx)
                        predictions.append((item_idx, pred_rating))
                    except:
                        continue
                
                predictions.sort(key=lambda x: x[1], reverse=True)
                pred_items = [item_idx for item_idx, _ in predictions]
            
            y_true.append(relevant_items)
            y_pred.append(pred_items)
        
        # Calculate metrics for each k value
        results = {}
        for k in k_values:
            results[f'k={k}'] = {
                'Precision@K': self.calculate_precision_at_k(y_true, y_pred, k),
                'Recall@K': self.calculate_recall_at_k(y_true, y_pred, k),
                'F1@K': self.calculate_f1_at_k(y_true, y_pred, k),
                'NDCG@K': self.calculate_ndcg_at_k(y_true, y_pred, k),
                'Hit_Rate@K': self.calculate_hit_rate_at_k(y_true, y_pred, k)
            }
        
        return results
    
    def evaluate_model(self, model: Any, test_data: pd.DataFrame, 
                      user_to_idx: Dict, item_to_idx: Dict,
                      k_values: List[int] = [5, 10, 20],
                      rating_threshold: float = 4.0) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained recommendation model
            test_data: Test data with columns ['user_idx', 'item_idx', 'rating']
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
            k_values: List of k values for ranking evaluation
            rating_threshold: Threshold for considering an item as relevant
            
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Evaluating model: {model.__class__.__name__}")
        
        results = {}
        
        # Rating prediction evaluation
        try:
            rating_metrics = self.evaluate_rating_prediction(model, test_data, user_to_idx, item_to_idx)
            results['rating_prediction'] = rating_metrics
        except Exception as e:
            print(f"Rating prediction evaluation failed: {e}")
            results['rating_prediction'] = {'error': str(e)}
        
        # Ranking evaluation
        try:
            ranking_metrics = self.evaluate_ranking(model, test_data, user_to_idx, item_to_idx, 
                                                  k_values, rating_threshold)
            results['ranking'] = ranking_metrics
        except Exception as e:
            print(f"Ranking evaluation failed: {e}")
            results['ranking'] = {'error': str(e)}
        
        # Model info
        try:
            model_info = model.get_model_info()
            results['model_info'] = model_info
        except:
            results['model_info'] = {'error': 'Could not retrieve model info'}
        
        return results
    
    def compare_models(self, models: Dict[str, Any], test_data: pd.DataFrame,
                      user_to_idx: Dict, item_to_idx: Dict,
                      k_values: List[int] = [5, 10, 20],
                      rating_threshold: float = 4.0) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            test_data: Test data
            user_to_idx: Mapping from user_id to user_index
            item_to_idx: Mapping from item_id to item_index
            k_values: List of k values for evaluation
            rating_threshold: Threshold for considering an item as relevant
            
        Returns:
            DataFrame with comparison results
        """
        print("Comparing models...")
        
        all_results = []
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            results = self.evaluate_model(model, test_data, user_to_idx, item_to_idx, 
                                        k_values, rating_threshold)
            
            # Flatten results for DataFrame
            flattened = {'model': model_name}
            
            # Add rating prediction metrics
            if 'rating_prediction' in results and 'error' not in results['rating_prediction']:
                for metric, value in results['rating_prediction'].items():
                    flattened[f'rating_{metric}'] = value
            
            # Add ranking metrics
            if 'ranking' in results and 'error' not in results['ranking']:
                for k_key, k_metrics in results['ranking'].items():
                    for metric, value in k_metrics.items():
                        flattened[f'{k_key}_{metric}'] = value
            
            all_results.append(flattened)
        
        return pd.DataFrame(all_results)
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Rating prediction results
        if 'rating_prediction' in results:
            print("\nRating Prediction Metrics:")
            print("-" * 30)
            if 'error' in results['rating_prediction']:
                print(f"Error: {results['rating_prediction']['error']}")
            else:
                for metric, value in results['rating_prediction'].items():
                    print(f"{metric}: {value:.4f}")
        
        # Ranking results
        if 'ranking' in results:
            print("\nRanking Metrics:")
            print("-" * 30)
            if 'error' in results['ranking']:
                print(f"Error: {results['ranking']['error']}")
            else:
                for k_key, k_metrics in results['ranking'].items():
                    print(f"\n{k_key}:")
                    for metric, value in k_metrics.items():
                        print(f"  {metric}: {value:.4f}")
        
        # Model info
        if 'model_info' in results:
            print("\nModel Information:")
            print("-" * 30)
            if 'error' in results['model_info']:
                print(f"Error: {results['model_info']['error']}")
            else:
                for key, value in results['model_info'].items():
                    print(f"{key}: {value}")

def main():
    """Main function to demonstrate evaluation framework"""
    evaluator = RecommenderEvaluator()
    
    # Example usage
    print("Recommender System Evaluation Framework")
    print("This module provides comprehensive evaluation metrics for recommendation systems.")
    print("\nAvailable metrics:")
    print("- Rating Prediction: RMSE, MAE")
    print("- Ranking: Precision@K, Recall@K, F1@K, NDCG@K, Hit Rate@K")
    print("\nUse the RecommenderEvaluator class to evaluate your models.")

if __name__ == "__main__":
    main()
