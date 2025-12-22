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


# ---------- REALISTIC MOCK DATA (MATCHING SCREENSHOT CATEGORIES) ----------

# 1. MOVIE DATA POOLS (Matching your UI buttons exactly)
MOVIES_BY_GENRE = {
    "Action": [
        "John Wick", "Mad Max: Fury Road", "Gladiator", "Die Hard", 
        "The Terminator", "Top Gun: Maverick", "The Dark Knight", "Taken"
    ],
    "Adventure": [
        "The Lord of the Rings", "Indiana Jones", "The Revenant", "Life of Pi", 
        "Into the Wild", "Cast Away", "Uncharted", "The Mummy"
    ],
    "Comedy": [
        "Superbad", "The Hangover", "Step Brothers", "Mean Girls", 
        "Dumb and Dumber", "Anchorman", "Bridesmaids", "Ferris Bueller's Day Off"
    ],
    "Drama": [
        "The Shawshank Redemption", "Forrest Gump", "The Godfather", "Fight Club", 
        "Good Will Hunting", "A Star Is Born", "Parasite", "Moonlight"
    ],
    "Thriller": [
        "Se7en", "Shutter Island", "Gone Girl", "Prisoners", 
        "Zodiac", "The Silence of the Lambs", "Black Swan", "Joker"
    ],
    "Horror": [
        "The Conjuring", "It", "Get Out", "Hereditary", 
        "A Quiet Place", "Halloween", "The Exorcist", "Us"
    ],
    "Romance": [
        "The Notebook", "La La Land", "Pride & Prejudice", "Titanic", 
        "Before Sunrise", "About Time", "The Fault in Our Stars", "Crazy Rich Asians"
    ],
    "Science Fiction": [
        "Inception", "Interstellar", "The Matrix", "Blade Runner 2049", 
        "Dune", "Arrival", "Ex Machina", "Star Wars: A New Hope"
    ],
    "Fantasy": [
        "Harry Potter", "Pan's Labyrinth", "The Chronicles of Narnia", "Spirited Away", 
        "The Wizard of Oz", "How to Train Your Dragon", "Alice in Wonderland"
    ],
    "Mystery": [
        "Knives Out", "Sherlock Holmes", "Memento", "Murder on the Orient Express", 
        "The Girl with the Dragon Tattoo", "Glass Onion", "Rear Window"
    ],
    "Crime": [
        "Pulp Fiction", "Goodfellas", "The Wolf of Wall Street", "City of God", 
        "The Departed", "Scarface", "Reservoir Dogs", "Heat"
    ]
}

# 2. BOOK DATA POOLS
BOOKS_BY_GENRE = {
    "Fiction": [
        "To Kill a Mockingbird", "The Great Gatsby", "Beloved", "The Kite Runner", 
        "Life of Pi", "The Alchemist", "The Book Thief", "Little Fires Everywhere"
    ],
    "Non-Fiction": [
        "Sapiens", "Educated", "Becoming", "The Diary of a Young Girl", 
        "Hiroshima", "In Cold Blood", "Into Thin Air", "Born a Crime"
    ],
    "Mystery": [
        "The Da Vinci Code", "Gone Girl", "Big Little Lies", "The Silent Patient", 
        "And Then There Were None", "The Girl on the Train", "Sharp Objects"
    ],
    "Fantasy": [
        "A Game of Thrones", "The Hobbit", "The Name of the Wind", "Percy Jackson", 
        "Mistborn", "American Gods", "Circe", "The Way of Kings"
    ],
    "Science Fiction": [
        "1984", "Dune", "Fahrenheit 451", "Ender's Game", 
        "The Martian", "Ready Player One", "Brave New World", "Neuromancer"
    ],
    "Romance": [
        "Pride and Prejudice", "Me Before You", "Outlander", "The Hating Game", 
        "It Ends with Us", "Jane Eyre", "Red, White & Royal Blue"
    ],
    "Historical": [
        "The Nightingale", "All the Light We Cannot See", "Wolf Hall", 
        "The Other Boleyn Girl", "Pillars of the Earth", "War and Peace"
    ],
    "Biography": [
        "Steve Jobs", "Elon Musk", "The Wright Brothers", "Shoe Dog", 
        "I Am Malala", "A Promised Land", "Alexander Hamilton"
    ],
    "Self-Help": [
        "Atomic Habits", "The Power of Habit", "The Subtle Art of Not Giving a F*ck", 
        "Deep Work", "Grit", "Mindset", "How to Win Friends and Influence People"
    ],
    "Poetry": [
        "Milk and Honey", "The Sun and Her Flowers", "Leaves of Grass", 
        "The Odyssey", "Paradise Lost", "Ariel", "The Waste Land"
    ]
}

# 3. PRODUCT DATA POOLS
PRODUCTS_BY_CATEGORY = {
    "Electronics": [
        "Wireless Noise Cancelling Headphones", "Smartphone 5G", "4K Ultra HD Smart TV", 
        "Gaming Laptop", "Bluetooth Speaker", "Digital Camera", "Smartwatch", "Tablet Pro"
    ],
    "Automotive": [
        "Car Vacuum Cleaner", "Dashboard Camera", "Car Seat Organizer", "Microfiber Cleaning Cloths", 
        "Windshield Sun Shade", "Tire Inflator", "Car Phone Mount", "LED Headlight Bulbs"
    ],
    "Sports and Outdoors": [
        "Yoga Mat", "Dumbbell Set", "Camping Tent", "Hiking Backpack", 
        "Running Shoes", "Fitness Tracker", "Insulated Water Bottle", "Fishing Rod"
    ],
    "Toys and Games": [
        "LEGO Set", "Board Game Classics", "Remote Control Car", "Action Figure", 
        "Puzzle 1000 Pieces", "Plush Teddy Bear", "Educational Science Kit", "Drone"
    ],
    "Home and Kitchen": [
        "Air Fryer", "Coffee Maker", "Robot Vacuum", "Blender", 
        "Memory Foam Pillow", "Non-Stick Cookware Set", "Electric Kettle", "Throw Blanket"
    ],
    "Beauty": [
        "Facial Moisturizer", "Vitamin C Serum", "Hair Dryer", "Makeup Brush Set", 
        "Exfoliating Scrub", "Sunscreen SPF 50", "Perfume", "Electric Shaver"
    ],
    "Musical Instruments": [
        "Acoustic Guitar", "Electric Keyboard", "Ukulele", "Drum Set", 
        "Violin", "Microphone Stand", "Guitar Tuner", "Harmonica"
    ]
}

# 4. GENERATE MOCK DATA STRUCTURE
mock_data = {
    "movies": [],
    "books": [],
    "products": []
}

# Helper to populate lists
def populate_dataset(source_dict, target_list, id_start, type_label):
    current_id = id_start
    # First pass: add all real items
    for category, titles in source_dict.items():
        for title in titles:
            item = {
                "item_id": current_id,
                "title": title,
            }
            # Handle different field names for categories
            if type_label == "movie":
                item["genres"] = category  # Assign the exact category name as genre
            elif type_label == "book":
                item["genres"] = category
                item["authors"] = "Best Selling Author"
            elif type_label == "product":
                item["category"] = category
            
            target_list.append(item)
            current_id += 1

    # Second pass: Fill to 500 items if necessary by duplicating with "Vol. 2" etc
    original_items = list(target_list) # Copy of the real items
    while len(target_list) < 500:
        base_item = original_items[len(target_list) % len(original_items)]
        new_item = base_item.copy()
        new_item["item_id"] = current_id
        new_item["title"] = f"{base_item['title']} (Special Edition)"
        target_list.append(new_item)
        current_id += 1

# Execute generation
populate_dataset(MOVIES_BY_GENRE, mock_data["movies"], 1, "movie")
populate_dataset(BOOKS_BY_GENRE, mock_data["books"], 1, "book")
populate_dataset(PRODUCTS_BY_CATEGORY, mock_data["products"], 1, "product")


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
        'songs': 'products'
    }

    mock_key = dataset_mapping.get(dataset, 'movies')
    mock_items = mock_data.get(mock_key, mock_data['movies'])

    # 1. Strict Filtering
    if categories and len(categories) > 0:
        filtered_items = []
        for item in mock_items:
            # Check for genre/category match
            item_cats = []
            if 'genres' in item:
                # Handle "Action|Thriller" vs "Action"
                item_cats = [g.strip() for g in item['genres'].split('|')]
            elif 'category' in item:
                item_cats = [str(item['category'])]
            
            # If the item matches ANY of the requested categories, keep it
            if any(cat in item_cats for cat in categories):
                filtered_items.append(item)
        
        # If we found matches, use ONLY those. 
        # If no matches found (rare with mock data), fall back to all items.
        if filtered_items:
            candidate_pool = filtered_items
        else:
            candidate_pool = mock_items
    else:
        # No categories requested? Use everything.
        candidate_pool = mock_items

    # 2. Randomize Selection from the filtered pool
    # Ensure we don't try to sample more items than exist
    sample_size = min(n_recommendations, len(candidate_pool))
    recommendations = random.sample(candidate_pool, sample_size)

    result = []
    for item in recommendations:
        result.append({
            'item_id': int(item['item_id']),
            'predicted_rating': round(random.uniform(3.5, 5.0), 2),
            'item_info': item
        })

    return {
        'user_id': int(user_id),
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
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations, categories=categories))

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
                return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations, categories=categories))
        else:
            user_idx = data['user_to_idx'][user_id]

        # Get all recommendations
        try:
            all_recommendations = model.recommend(user_idx, n_recommendations * 5)  # Get more to filter
        except Exception as e:
            print(f"Error getting recommendations: {e}, using fallback")
            return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations, categories=categories))

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
        return jsonify(get_fallback_recommendations(dataset, user_id, n_recommendations, categories=categories))

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