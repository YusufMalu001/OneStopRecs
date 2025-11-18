# 🎬📚🛍️ Full-Stack Recommender System

## 🎯 Project Overview
This project implements a complete full-stack recommender system with a React frontend and Python Flask backend. It provides an interactive web interface for users to get personalized recommendations using machine learning models trained on public benchmark datasets.

## 🏗️ Architecture
- **Frontend**: React.js with modern UI components
- **Backend**: Flask API serving machine learning models
- **Models**: Collaborative Filtering, Matrix Factorization, Neural CF
- **Datasets**: MovieLens, GoodBooks, Amazon Reviews

## 🚀 Quick Start

### 1. Start the Backend API
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python src/api_server.py
```

### 2. Start the Frontend
```bash
# Install frontend dependencies
cd frontend
npm install

# Start the React app
npm start
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000

## 📊 Datasets Used
1. **MovieLens 100K** - Movie ratings (943 users, 1,682 movies, 100K ratings)
2. **GoodBooks-10K** - Book ratings (53K users, 10K books, 6M ratings)
3. **Amazon Reviews** - E-commerce products (10K users, 5K products, 50K ratings)

## 🤖 Models Implemented
1. **Collaborative Filtering** - User-based and Item-based approaches
2. **Matrix Factorization** - SVD and NMF techniques
3. **Neural Collaborative Filtering** - Deep learning approach

## 🎨 Frontend Features
- ✅ Interactive category selection (Movies, Books, Products)
- ✅ Multi-tag selection with visual feedback
- ✅ Real-time personalized recommendations
- ✅ Responsive design for all devices
- ✅ Loading states and error handling
- ✅ Model information and confidence scores

## 🔧 Backend Features
- ✅ RESTful API with comprehensive endpoints
- ✅ Multiple ML models serving recommendations
- ✅ Category-based filtering
- ✅ Search functionality
- ✅ Rating prediction
- ✅ Health monitoring and status endpoints

## 📁 Project Structure
```
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API service layer
│   │   └── config.js      # Configuration
│   └── package.json
├── src/                   # Backend source code
│   ├── api_server.py     # Flask API server
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models/           # Model implementations
│   ├── evaluation.py     # Evaluation metrics
│   └── main.py          # Standalone experiment runner
├── data/                 # Dataset storage
├── results/              # Results and visualizations
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🔄 How It Works

1. **User selects a category** (Movies, Books, or Products)
2. **User chooses relevant tags/genres** from available options
3. **Frontend sends API request** to backend with user preferences
4. **Backend processes request** using trained ML models
5. **Recommendations are returned** and displayed in an attractive UI
6. **User can explore results** with ratings and confidence scores

## 📈 Evaluation Metrics
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Precision@K, Recall@K, F1@K**
- **NDCG@K** (Normalized Discounted Cumulative Gain)
- **Hit Rate@K**

## 🛠️ Development

### Backend Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode with auto-reload
python src/api_server.py
```

### Frontend Development
```bash
cd frontend
npm install
npm start  # Auto-reloads on changes
```

### Running Experiments
```bash
# Run standalone experiments (without web interface)
python src/main.py
```

## 📚 Documentation
- **README.md** - Complete setup and usage guide
- **Source Code** - Well-documented Python and React code

## 🎯 Key Features

### Interactive Web Interface
- Modern, responsive React frontend
- Real-time recommendation updates
- Category and tag-based filtering
- Beautiful result visualization

### Robust Backend API
- Flask REST API with comprehensive endpoints
- Multiple ML models serving recommendations
- Error handling and validation
- Health monitoring and status checks

### Comprehensive Evaluation
- Multiple evaluation metrics
- Cross-dataset performance comparison
- Model information and statistics
- Reproducible experiments

## 🚀 Production Ready
- Modular, well-documented code
- Error handling and logging
- CORS configuration for web integration
- Scalable architecture
- Docker-ready deployment

---

**🎉 This project demonstrates a complete, production-ready recommender system that combines modern web technologies with advanced machine learning algorithms!**
