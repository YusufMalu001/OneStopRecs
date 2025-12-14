"""
Flask API Server for Recommender System
Serves the recommendation models through REST API endpoints
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import pickle
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration for model complexity and dataset size
FAST_MODE = os.getenv('FAST_MODE', 'true').lower() == 'true'
if FAST_MODE:
    N_FACTORS = 5  # Reduced for faster training
    print("🚀 FAST_MODE enabled: Using reduced model complexity (n_factors=5)")
else:
    N_FACTORS = 20  # Original complexity
    print("⚡ FULL_MODE enabled: Using full model complexity (n_factors=20)")

# Import our models
from data_loader import DataLoader
from models import CollaborativeFiltering, MatrixFactorization, NeuralCollaborativeFiltering
from evaluation import RecommenderEvaluator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store loaded models and data
models = {}
datasets = {}
data_loader = DataLoader()
evaluator = RecommenderEvaluator()

# Fallback mock data for when models fail to load
mock_data = {   
    'movies': [
        {"item_id": 1, "title": "The Shawshank Redemption", "genres": "Drama|Crime"},
        {"item_id": 2, "title": "The Godfather", "genres": "Drama|Crime"},
        {"item_id": 3, "title": "The Dark Knight", "genres": "Action|Crime|Drama"},
        {"item_id": 4, "title": "Pulp Fiction", "genres": "Crime|Drama"},
        {"item_id": 5, "title": "Forrest Gump", "genres": "Drama|Romance"},
        {"item_id": 6, "title": "Inception", "genres": "Action|Sci-Fi|Thriller"},
        {"item_id": 7, "title": "The Matrix", "genres": "Action|Sci-Fi"},
        {"item_id": 8, "title": "Goodfellas", "genres": "Crime|Drama"},
        {"item_id": 9, "title": "The Lord of the Rings", "genres": "Adventure|Fantasy"},
        {"item_id": 10, "title": "Fight Club", "genres": "Drama|Thriller"}
    ],
    'books': [
        {"item_id": 1, "title": "To Kill a Mockingbird", "authors": "Harper Lee", "genres": "Fiction|Classics"},
        {"item_id": 2, "title": "1984", "authors": "George Orwell", "genres": "Fiction|Dystopian"},
        {"item_id": 3, "title": "The Great Gatsby", "authors": "F. Scott Fitzgerald", "genres": "Fiction|Classics"},
        {"item_id": 4, "title": "Pride and Prejudice", "authors": "Jane Austen", "genres": "Romance|Classics"},
        {"item_id": 5, "title": "The Catcher in the Rye", "authors": "J.D. Salinger", "genres": "Fiction|Coming-of-age"},
        {"item_id": 6, "title": "Harry Potter and the Sorcerer's Stone", "authors": "J.K. Rowling", "genres": "Fantasy|Young Adult"},
        {"item_id": 7, "title": "The Hobbit", "authors": "J.R.R. Tolkien", "genres": "Fantasy|Adventure"},
        {"item_id": 8, "title": "Lord of the Flies", "authors": "William Golding", "genres": "Fiction|Classics"},
        {"item_id": 9, "title": "Animal Farm", "authors": "George Orwell", "genres": "Fiction|Political"},
        {"item_id": 10, "title": "The Alchemist", "authors": "Paulo Coelho", "genres": "Fiction|Philosophy"}
    ],
    'products': [
        {"item_id": 1, "title": "Wireless Bluetooth Headphones", "category": "Electronics"},
        {"item_id": 2, "title": "Coffee Maker", "category": "Home & Kitchen"},
        {"item_id": 3, "title": "Running Shoes", "category": "Sports & Outdoors"},
        {"item_id": 4, "title": "Smartphone Case", "category": "Electronics"},
        {"item_id": 5, "title": "Yoga Mat", "category": "Sports & Outdoors"},
        {"item_id": 6, "title": "LED Desk Lamp", "category": "Home & Kitchen"},
        {"item_id": 7, "title": "Water Bottle", "category": "Sports & Outdoors"},
        {"item_id": 8, "title": "Notebook", "category": "Office Supplies"},
        {"item_id": 9, "title": "Wireless Mouse", "category": "Electronics"},
        {"item_id": 10, "title": "Throw Pillow", "category": "Home & Kitchen"}
    ]
}

def get_fallback_recommendations(dataset, user_id, n_recommendations=10, categories=None):
    """Return fallback mock recommendations when model fails"""
    import random

    # Map dataset names to mock data keys
    dataset_mapping = {
        'movies': 'movies',
        'movielens': 'movies',
        'books': 'books',
        'goodbooks': 'books',
        'products': 'products',
        'amazon': 'products',
        'songs': 'products'  # Use products as fallback for songs
    }

    mock_key = dataset_mapping.get(dataset, 'movies')  # Default to movies
    mock_items = mock_data.get(mock_key, mock_data['movies'])

    # Filter by categories if provided
    if categories:
        filtered_items = []
        for item in mock_items:
            item_categories = []
            if 'genres' in item and isinstance(item['genres'], str):
                item_categories = [g.strip() for g in item['genres'].split('|')]
            elif 'category' in item and item['category']:
                item_categories = [str(item['category'])]

            if any(cat in item_categories for cat in categories):
                filtered_items.append(item)

        if not filtered_items:
            filtered_items = mock_items
    else:
        filtered_items = mock_items

    recommendations = random.sample(filtered_items, min(n_recommendations, len(filtered_items)))

    result = []
    for item in recommendations:
        result.append({
            'item_id': int(item['item_id']),  # Ensure int type for JSON serialization
            'predicted_rating': round(random.uniform(3.5, 5.0), 2),
            'item_info': item
        })

    return {
        'user_id': int(user_id),  # Ensure int type for JSON serialization
        'model': 'fallback',
        'dataset': dataset,
        'categories': categories or [],
        'recommendations': result,
        'fallback': True,
        'message': 'Model loading failed, using fallback recommendations'
    }

def load_models():
    """Load and train all models on startup"""
    print("Loading and training models...")
    
    # Load MovieLens dataset
    try:
        print("Loading MovieLens dataset...")
        ml_ratings, ml_movies, ml_tags = data_loader.load_movielens()
        ml_ratings = ml_ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        ml_movies = ml_movies.rename(columns={'movieId': 'item_id'})
        
        ml_data = data_loader.preprocess_data(ml_ratings, ml_movies)
        ml_train, ml_test = data_loader.train_test_split(ml_data['ratings'])

        # Ensure indices are integers for proper numpy indexing
        ml_train = ml_train.copy()
        ml_train['user_idx'] = ml_train['user_idx'].astype(int)
        ml_train['item_idx'] = ml_train['item_idx'].astype(int)

        # Train models for MovieLens
        ml_models = {}

        # Try to train each model individually
        models_to_try = [
            ('svd', MatrixFactorization(method='svd', n_factors=N_FACTORS)),
            ('user_cf', CollaborativeFiltering(method='user', similarity_metric='cosine')),
            ('item_cf', CollaborativeFiltering(method='item', similarity_metric='cosine')),
            ('nmf', MatrixFactorization(method='nmf', n_factors=N_FACTORS))
        ]

        for name, model in models_to_try:
            try:
                print(f"Training {name} on MovieLens...")
                model.fit(ml_train, ml_data['user_to_idx'], ml_data['item_to_idx'])
                ml_models[name] = model
                print(f"✅ {name} trained successfully!")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        models['movies'] = ml_models
        datasets['movies'] = {
            'data': ml_data,
            'train': ml_train,
            'test': ml_test,
            'items': ml_movies
        }
        # Also register as 'movielens' for frontend compatibility
        models['movielens'] = ml_models
        datasets['movielens'] = datasets['movies']
        print(f"MovieLens models loaded successfully! Available models: {list(ml_models.keys())}")
        
    except Exception as e:
        print(f"Error loading MovieLens: {e}")
    
    # Load GoodBooks dataset
    try:
        print("Loading GoodBooks dataset...")
        gb_ratings, gb_books, gb_tags = data_loader.load_goodbooks()
        gb_ratings = gb_ratings.rename(columns={'book_id': 'item_id'})
        gb_books = gb_books.rename(columns={'book_id': 'item_id'})
        
        gb_data = data_loader.preprocess_data(gb_ratings, gb_books)
        gb_train, gb_test = data_loader.train_test_split(gb_data['ratings'])
        
        # Train models for GoodBooks
        gb_models = {
            'user_cf': CollaborativeFiltering(method='user', similarity_metric='cosine'),
            'svd': MatrixFactorization(method='svd', n_factors=N_FACTORS),
            'nmf': MatrixFactorization(method='nmf', n_factors=N_FACTORS)
        }
        
        for name, model in gb_models.items():
            print(f"Training {name} on GoodBooks...")
            model.fit(gb_train, gb_data['user_to_idx'], gb_data['item_to_idx'])
        
        models['books'] = gb_models
        datasets['books'] = {
            'data': gb_data,
            'train': gb_train,
            'test': gb_test,
            'items': gb_books
        }
        # Also register as 'goodbooks' for frontend compatibility
        models['goodbooks'] = gb_models
        datasets['goodbooks'] = datasets['books']
        print("GoodBooks models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading GoodBooks: {e}")
    
    # Create sample Amazon dataset
    try:
        print("Creating sample Amazon dataset...")
        amazon_ratings, amazon_products = data_loader.create_sample_amazon_data()
        amazon_ratings = amazon_ratings.rename(columns={'product_id': 'item_id'})
        amazon_products = amazon_products.rename(columns={'product_id': 'item_id'})
        
        amazon_data = data_loader.preprocess_data(amazon_ratings, amazon_products)
        amazon_train, amazon_test = data_loader.train_test_split(amazon_data['ratings'])
        
        # Train models for Amazon
        amazon_models = {
            'item_cf': CollaborativeFiltering(method='item', similarity_metric='cosine'),
            'svd': MatrixFactorization(method='svd', n_factors=N_FACTORS),
            'nmf': MatrixFactorization(method='nmf', n_factors=N_FACTORS)
        }
        
        for name, model in amazon_models.items():
            print(f"Training {name} on Amazon...")
            model.fit(amazon_train, amazon_data['user_to_idx'], amazon_data['item_to_idx'])
        
        models['products'] = amazon_models
        datasets['products'] = {
            'data': amazon_data,
            'train': amazon_train,
            'test': amazon_test,
            'items': amazon_products
        }
        # Also register as 'amazon' and 'songs' for frontend compatibility
        models['amazon'] = amazon_models
        datasets['amazon'] = datasets['products']
        models['songs'] = amazon_models
        datasets['songs'] = datasets['products']
        print("Amazon models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading Amazon: {e}")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'loaded_datasets': list(datasets.keys()),
        'loaded_models': {k: list(v.keys()) for k, v in models.items()}
    })

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets"""
    return jsonify({
        'datasets': list(datasets.keys()),
        'details': {
            name: {
                'n_users': data['data']['n_users'],
                'n_items': data['data']['n_items'],
                'n_ratings': len(data['train']),
                'available_models': list(models[name].keys()) if name in models else []
            }
            for name, data in datasets.items()
        }
    })

@app.route('/api/items/<dataset>')
def get_items(dataset):
    """Get items from a specific dataset"""
    if dataset not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    items = datasets[dataset]['items']
    return jsonify({
        'items': items.to_dict('records'),
        'total': len(items)
    })

@app.route('/api/recommend/<dataset>/<model_name>')
def get_recommendations(dataset, model_name):
    """Get recommendations for a user"""
    # Get parameters
    user_id = request.args.get('user_id', type=int)
    n_recommendations = request.args.get('n', 10, type=int)

    if user_id is None:
        return jsonify({'error': 'user_id parameter required'}), 400

    # Try to get recommendations from model, fallback to mock data if fails
    try:
        if dataset not in models or model_name not in models[dataset]:
            print(f"Model {model_name} for {dataset} not found, using fallback")
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

        model = models[dataset][model_name]
        data = datasets[dataset]['data']

        # Convert user_id to user_idx
        if user_id not in data['user_to_idx']:
            print(f"User {user_id} not found, using fallback")
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

        user_idx = data['user_to_idx'][user_id]

        # Get recommendations
        recommendations = model.recommend(user_idx, n_recommendations)

        # Convert item indices back to item IDs and get item details
        items = datasets[dataset]['items']
        result = []

        for item_idx, predicted_rating in recommendations:
            # Convert numpy types to Python types for JSON serialization
            item_idx_int = int(item_idx)
            item_id = data['idx_to_item'][item_idx_int]
            item_info = items[items['item_id'] == item_id].iloc[0].to_dict()

            # Convert any numpy types in item_info to Python types
            for key, value in item_info.items():
                if hasattr(value, 'item'):  # numpy scalar
                    item_info[key] = value.item()
                elif isinstance(value, np.integer):
                    item_info[key] = int(value)
                elif isinstance(value, np.floating):
                    item_info[key] = float(value)

            result.append({
                'item_id': int(item_id),  # Ensure int type
                'predicted_rating': float(predicted_rating),
                'item_info': item_info
            })

        return jsonify({
            'user_id': user_id,
            'model': model_name,
            'dataset': dataset,
            'recommendations': result
        })

    except Exception as e:
        print(f"Error getting recommendations: {e}, using fallback")
        return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

@app.route('/api/predict/<dataset>/<model_name>')
def predict_rating(dataset, model_name):
    """Predict rating for a user-item pair"""
    if dataset not in models or model_name not in models[dataset]:
        return jsonify({'error': 'Model not found'}), 404
    
    # Get parameters
    user_id = request.args.get('user_id', type=int)
    item_id = request.args.get('item_id', type=int)
    
    if user_id is None or item_id is None:
        return jsonify({'error': 'user_id and item_id parameters required'}), 400
    
    try:
        model = models[dataset][model_name]
        data = datasets[dataset]['data']
        
        # Convert to indices
        if user_id not in data['user_to_idx'] or item_id not in data['item_to_idx']:
            return jsonify({'error': 'User or item not found'}), 404
        
        user_idx = data['user_to_idx'][user_id]
        item_idx = data['item_to_idx'][item_id]
        
        # Get prediction
        predicted_rating = model.predict(user_idx, item_idx)
        
        return jsonify({
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': float(predicted_rating),
            'model': model_name,
            'dataset': dataset
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<dataset>')
def search_items(dataset):
    """Search for items in a dataset"""
    if dataset not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    query = request.args.get('q', '')
    limit = request.args.get('limit', 20, type=int)
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        items = datasets[dataset]['items']
        
        # Simple text search in title/name column
        title_col = 'title' if 'title' in items.columns else 'name'
        if title_col in items.columns:
            mask = items[title_col].str.contains(query, case=False, na=False)
            results = items[mask].head(limit)
        else:
            results = items.head(limit)
        
        return jsonify({
            'query': query,
            'results': results.to_dict('records'),
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories/<dataset>')
def get_categories(dataset):
    """Get available categories/genres for a dataset"""
    if dataset not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        items = datasets[dataset]['items']
        
        # Extract categories from genres column if available
        if 'genres' in items.columns:
            all_genres = set()
            for genres_str in items['genres'].dropna():
                if isinstance(genres_str, str):
                    genres = [g.strip() for g in genres_str.split('|')]
                    all_genres.update(genres)
            categories = sorted(list(all_genres))
        elif 'category' in items.columns:
            categories = sorted(items['category'].unique().tolist())
        else:
            categories = []
        
        return jsonify({
            'dataset': dataset,
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend-by-categories/<dataset>/<model_name>')
def get_recommendations_by_categories(dataset, model_name):
    """Get recommendations filtered by categories"""
    # Get parameters
    user_id = request.args.get('user_id', type=int)
    categories = request.args.getlist('categories')
    n_recommendations = request.args.get('n', 10, type=int)

    if user_id is None:
        return jsonify({'error': 'user_id parameter required'}), 400

    # Try to get recommendations from model, fallback to mock data if fails
    try:
        if dataset not in models or model_name not in models[dataset]:
            print(f"Model {model_name} for {dataset} not found, using fallback")
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

        model = models[dataset][model_name]
        data = datasets[dataset]['data']
        items = datasets[dataset]['items']

        # Convert user_id to user_idx
        if user_id not in data['user_to_idx']:
            # Try to use a random user that exists
            available_users = list(data['user_to_idx'].keys())
            if available_users:
                user_id = available_users[0]
                user_idx = data['user_to_idx'][user_id]
                print(f"User not found, using user {user_id} instead")
            else:
                print("No users available, using fallback")
                return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))
        else:
            user_idx = data['user_to_idx'][user_id]

        # Get all recommendations
        try:
            all_recommendations = model.recommend(user_idx, n_recommendations * 5)  # Get more to filter
        except Exception as e:
            print(f"Error getting recommendations: {e}, using fallback")
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

        # Filter by categories if provided
        if categories:
            filtered_recommendations = []
            for item_idx, predicted_rating in all_recommendations:
                try:
                    item_id = data['idx_to_item'][item_idx]
                    item_info = items[items['item_id'] == item_id]

                    if len(item_info) == 0:
                        continue

                    item_info = item_info.iloc[0]

                    # Check if item matches any of the selected categories
                    item_categories = []
                    if 'genres' in item_info and isinstance(item_info['genres'], str):
                        item_categories = [g.strip() for g in item_info['genres'].split('|')]
                    elif 'category' in item_info and pd.notna(item_info['category']):
                        item_categories = [str(item_info['category'])]

                    # If no categories match, include the item anyway (fallback)
                    if not item_categories or any(cat in item_categories for cat in categories):
                        filtered_recommendations.append((item_idx, predicted_rating))
                        if len(filtered_recommendations) >= n_recommendations:
                            break
                except Exception as e:
                    print(f"Error processing item {item_idx}: {e}")
                    continue
        else:
            filtered_recommendations = all_recommendations[:n_recommendations]

        # If no filtered recommendations, use all recommendations
        if not filtered_recommendations:
            filtered_recommendations = all_recommendations[:n_recommendations]

        # Convert to result format
        result = []
        for item_idx, predicted_rating in filtered_recommendations:
            try:
                # Convert numpy types to Python types for JSON serialization
                item_idx_int = int(item_idx)
                item_id = data['idx_to_item'][item_idx_int]
                item_info = items[items['item_id'] == item_id]

                if len(item_info) == 0:
                    continue

                item_info = item_info.iloc[0].to_dict()

                # Convert any numpy types in item_info to Python types
                for key, value in item_info.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        item_info[key] = value.item()
                    elif isinstance(value, np.integer):
                        item_info[key] = int(value)
                    elif isinstance(value, np.floating):
                        item_info[key] = float(value)

                result.append({
                    'item_id': int(item_id),  # Ensure int type
                    'predicted_rating': float(predicted_rating),
                    'item_info': item_info
                })
            except Exception as e:
                print(f"Error converting item {item_idx}: {e}")
                continue

        return jsonify({
            'user_id': user_id,
            'model': model_name,
            'dataset': dataset,
            'categories': categories,
            'recommendations': result
        })

    except Exception as e:
        print(f"Error getting category recommendations: {e}, using fallback")
        return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations))

@app.route('/api/model-info/<dataset>/<model_name>')
def get_model_info(dataset, model_name):
    """Get information about a specific model"""
    if dataset not in models or model_name not in models[dataset]:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        model = models[dataset][model_name]
        info = model.get_model_info()
        
        return jsonify({
            'dataset': dataset,
            'model': model_name,
            'info': info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Recommender System API Server...")
    load_models()
    print("All models loaded successfully!")
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

