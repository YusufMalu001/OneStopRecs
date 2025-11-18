"""
Main execution script for Recommender System Project
Runs experiments on 3 datasets with 3 different models
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from models import CollaborativeFiltering, MatrixFactorization, NeuralCollaborativeFiltering
from evaluation import RecommenderEvaluator

class RecommenderSystemExperiment:
    """Main experiment class for running recommender system comparisons"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.data_loader = DataLoader()
        self.evaluator = RecommenderEvaluator()
        self.ensure_results_dir()
        
    def ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def run_movielens_experiment(self) -> Dict[str, Any]:
        """Run experiment on MovieLens dataset"""
        print("\n" + "="*60)
        print("MOVIELENS DATASET EXPERIMENT")
        print("="*60)
        
        # Load and preprocess data
        ratings, movies, tags = self.data_loader.load_movielens()
        ratings = ratings.rename(columns={'movieId': 'item_id'})
        movies = movies.rename(columns={'movieId': 'item_id'})
        
        data = self.data_loader.preprocess_data(ratings, movies)
        train_data, test_data = self.data_loader.train_test_split(data['ratings'])
        
        print(f"Dataset stats: {len(train_data)} train, {len(test_data)} test samples")
        
        # Initialize models
        models = {
            'User-Based CF': CollaborativeFiltering(method='user', similarity_metric='cosine'),
            'Item-Based CF': CollaborativeFiltering(method='item', similarity_metric='cosine'),
            'SVD': MatrixFactorization(method='svd', n_factors=50),
            'NMF': MatrixFactorization(method='nmf', n_factors=50),
            'Neural CF': NeuralCollaborativeFiltering(n_factors=50, hidden_layers=[128, 64, 32])
        }
        
        # Train and evaluate models
        results = {}
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            try:
                if model_name == 'Neural CF':
                    # For neural model, use smaller subset for faster training
                    train_subset = train_data.sample(n=min(10000, len(train_data)), random_state=42)
                    model.fit(train_subset, data['user_to_idx'], data['item_to_idx'])
                else:
                    model.fit(train_data, data['user_to_idx'], data['item_to_idx'])
                
                # Evaluate model
                model_results = self.evaluator.evaluate_model(
                    model, test_data, data['user_to_idx'], data['item_to_idx']
                )
                results[model_name] = model_results
                
                # Print results
                self.evaluator.print_results(model_results)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def run_goodbooks_experiment(self) -> Dict[str, Any]:
        """Run experiment on GoodBooks dataset"""
        print("\n" + "="*60)
        print("GOODBOOKS DATASET EXPERIMENT")
        print("="*60)
        
        # Load and preprocess data
        ratings, books, book_tags = self.data_loader.load_goodbooks()
        ratings = ratings.rename(columns={'book_id': 'item_id'})
        books = books.rename(columns={'book_id': 'item_id'})
        
        data = self.data_loader.preprocess_data(ratings, books)
        train_data, test_data = self.data_loader.train_test_split(data['ratings'])
        
        print(f"Dataset stats: {len(train_data)} train, {len(test_data)} test samples")
        
        # Initialize models
        models = {
            'User-Based CF': CollaborativeFiltering(method='user', similarity_metric='cosine'),
            'SVD': MatrixFactorization(method='svd', n_factors=50),
            'Neural CF': NeuralCollaborativeFiltering(n_factors=50, hidden_layers=[128, 64, 32])
        }
        
        # Train and evaluate models
        results = {}
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            try:
                if model_name == 'Neural CF':
                    # For neural model, use smaller subset for faster training
                    train_subset = train_data.sample(n=min(15000, len(train_data)), random_state=42)
                    model.fit(train_subset, data['user_to_idx'], data['item_to_idx'])
                else:
                    model.fit(train_data, data['user_to_idx'], data['item_to_idx'])
                
                # Evaluate model
                model_results = self.evaluator.evaluate_model(
                    model, test_data, data['user_to_idx'], data['item_to_idx']
                )
                results[model_name] = model_results
                
                # Print results
                self.evaluator.print_results(model_results)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def run_amazon_experiment(self) -> Dict[str, Any]:
        """Run experiment on Amazon dataset"""
        print("\n" + "="*60)
        print("AMAZON DATASET EXPERIMENT")
        print("="*60)
        
        # Load and preprocess data
        ratings, products = self.data_loader.create_sample_amazon_data()
        ratings = ratings.rename(columns={'product_id': 'item_id'})
        products = products.rename(columns={'product_id': 'item_id'})
        
        data = self.data_loader.preprocess_data(ratings, products)
        train_data, test_data = self.data_loader.train_test_split(data['ratings'])
        
        print(f"Dataset stats: {len(train_data)} train, {len(test_data)} test samples")
        
        # Initialize models
        models = {
            'Item-Based CF': CollaborativeFiltering(method='item', similarity_metric='cosine'),
            'NMF': MatrixFactorization(method='nmf', n_factors=50),
            'Neural CF': NeuralCollaborativeFiltering(n_factors=50, hidden_layers=[128, 64, 32])
        }
        
        # Train and evaluate models
        results = {}
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            try:
                if model_name == 'Neural CF':
                    # For neural model, use smaller subset for faster training
                    train_subset = train_data.sample(n=min(10000, len(train_data)), random_state=42)
                    model.fit(train_subset, data['user_to_idx'], data['item_to_idx'])
                else:
                    model.fit(train_data, data['user_to_idx'], data['item_to_idx'])
                
                # Evaluate model
                model_results = self.evaluator.evaluate_model(
                    model, test_data, data['user_to_idx'], data['item_to_idx']
                )
                results[model_name] = model_results
                
                # Print results
                self.evaluator.print_results(model_results)
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def create_comparison_plots(self, all_results: Dict[str, Dict[str, Any]]):
        """Create comparison plots for all experiments"""
        print("\nCreating comparison plots...")
        
        # Extract metrics for plotting
        datasets = list(all_results.keys())
        metrics_data = []
        
        for dataset, results in all_results.items():
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    continue
                
                # Rating prediction metrics
                if 'rating_prediction' in model_results and 'error' not in model_results['rating_prediction']:
                    for metric, value in model_results['rating_prediction'].items():
                        metrics_data.append({
                            'Dataset': dataset,
                            'Model': model_name,
                            'Metric': f'Rating_{metric}',
                            'Value': value
                        })
                
                # Ranking metrics (use k=10 for comparison)
                if 'ranking' in model_results and 'error' not in model_results['ranking']:
                    if 'k=10' in model_results['ranking']:
                        for metric, value in model_results['ranking']['k=10'].items():
                            metrics_data.append({
                                'Dataset': dataset,
                                'Model': model_name,
                                'Metric': f'Ranking_{metric}',
                                'Value': value
                            })
        
        if not metrics_data:
            print("No valid results to plot")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recommender System Performance Comparison', fontsize=16)
        
        # Rating prediction metrics
        rating_metrics = df[df['Metric'].str.startswith('Rating_')]
        if not rating_metrics.empty:
            rating_pivot = rating_metrics.pivot_table(
                index=['Dataset', 'Model'], columns='Metric', values='Value'
            )
            
            # RMSE plot
            if 'Rating_RMSE' in rating_pivot.columns:
                rating_pivot['Rating_RMSE'].unstack().plot(kind='bar', ax=axes[0,0])
                axes[0,0].set_title('RMSE Comparison')
                axes[0,0].set_ylabel('RMSE')
                axes[0,0].tick_params(axis='x', rotation=45)
            
            # MAE plot
            if 'Rating_MAE' in rating_pivot.columns:
                rating_pivot['Rating_MAE'].unstack().plot(kind='bar', ax=axes[0,1])
                axes[0,1].set_title('MAE Comparison')
                axes[0,1].set_ylabel('MAE')
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # Ranking metrics
        ranking_metrics = df[df['Metric'].str.startswith('Ranking_')]
        if not ranking_metrics.empty:
            ranking_pivot = ranking_metrics.pivot_table(
                index=['Dataset', 'Model'], columns='Metric', values='Value'
            )
            
            # Precision@10 plot
            if 'Ranking_Precision@K' in ranking_pivot.columns:
                ranking_pivot['Ranking_Precision@K'].unstack().plot(kind='bar', ax=axes[1,0])
                axes[1,0].set_title('Precision@10 Comparison')
                axes[1,0].set_ylabel('Precision@10')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # NDCG@10 plot
            if 'Ranking_NDCG@K' in ranking_pivot.columns:
                ranking_pivot['Ranking_NDCG@K'].unstack().plot(kind='bar', ax=axes[1,1])
                axes[1,1].set_title('NDCG@10 Comparison')
                axes[1,1].set_ylabel('NDCG@10')
                axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plots saved to {self.results_dir}/performance_comparison.png")
    
    def save_results(self, all_results: Dict[str, Dict[str, Any]]):
        """Save results to files"""
        print("\nSaving results...")
        
        # Save detailed results as JSON
        import json
        with open(os.path.join(self.results_dir, 'detailed_results.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(all_results, f, indent=2, default=convert_numpy)
        
        # Create summary table
        summary_data = []
        for dataset, results in all_results.items():
            for model_name, model_results in results.items():
                if 'error' in model_results:
                    continue
                
                row = {'Dataset': dataset, 'Model': model_name}
                
                # Add rating prediction metrics
                if 'rating_prediction' in model_results and 'error' not in model_results['rating_prediction']:
                    for metric, value in model_results['rating_prediction'].items():
                        row[f'Rating_{metric}'] = value
                
                # Add ranking metrics (k=10)
                if 'ranking' in model_results and 'error' not in model_results['ranking']:
                    if 'k=10' in model_results['ranking']:
                        for metric, value in model_results['ranking']['k=10'].items():
                            row[f'Ranking_{metric}'] = value
                
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.results_dir, 'summary_results.csv'), index=False)
            print(f"Summary results saved to {self.results_dir}/summary_results.csv")
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("Starting Recommender System Experiments")
        print("="*60)
        
        all_results = {}
        
        # Run experiments on all datasets
        try:
            all_results['MovieLens'] = self.run_movielens_experiment()
        except Exception as e:
            print(f"MovieLens experiment failed: {e}")
            all_results['MovieLens'] = {'error': str(e)}
        
        try:
            all_results['GoodBooks'] = self.run_goodbooks_experiment()
        except Exception as e:
            print(f"GoodBooks experiment failed: {e}")
            all_results['GoodBooks'] = {'error': str(e)}
        
        try:
            all_results['Amazon'] = self.run_amazon_experiment()
        except Exception as e:
            print(f"Amazon experiment failed: {e}")
            all_results['Amazon'] = {'error': str(e)}
        
        # Create visualizations and save results
        self.create_comparison_plots(all_results)
        self.save_results(all_results)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*60)
        print(f"Results saved to {self.results_dir}/")
        
        return all_results

def main():
    """Main function"""
    experiment = RecommenderSystemExperiment()
    results = experiment.run_all_experiments()
    
    print("\nExperiment Summary:")
    for dataset, dataset_results in results.items():
        print(f"\n{dataset}:")
        for model, model_results in dataset_results.items():
            if 'error' in model_results:
                print(f"  {model}: ERROR - {model_results['error']}")
            else:
                print(f"  {model}: SUCCESS")

if __name__ == "__main__":
    main()
