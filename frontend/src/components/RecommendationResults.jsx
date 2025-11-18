import React, { useState, useEffect } from 'react';
import api from '../services/api';
import './RecommendationResults.css';

const RecommendationResults = ({ 
  dataset, 
  selectedTags, 
  userId = 1, 
  model = 'svd' 
}) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  // Map dataset names to API dataset names
  const datasetMap = {
    'Movies': 'movies',
    'Books': 'books',
    'Products': 'products' // Using products dataset for products
  };

  const modelMap = {
    'Movies': ['svd', 'nmf', 'user_cf', 'item_cf'],
    'Books': ['svd', 'nmf', 'user_cf'],
    'Products': ['svd', 'nmf', 'item_cf']
  };

  useEffect(() => {
    if (dataset && selectedTags.length > 0) {
      fetchRecommendations();
    }
  }, [dataset, selectedTags, userId, model]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const apiDataset = datasetMap[dataset];
      const availableModels = modelMap[dataset] || ['svd'];
      const selectedModel = availableModels.includes(model) ? model : availableModels[0];
      
      console.log('Fetching recommendations with:', {
        dataset: apiDataset,
        model: selectedModel,
        userId,
        selectedTags
      });
      
      // Get recommendations filtered by categories
      const response = await api.getRecommendationsByCategories(
        apiDataset,
        selectedModel,
        userId,
        selectedTags,
        10
      );
      
      console.log('API Response:', response);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      setRecommendations(response.recommendations || []);
      
      // Get model info
      try {
        const modelInfoResponse = await api.getModelInfo(apiDataset, selectedModel);
        setModelInfo(modelInfoResponse.info);
      } catch (infoError) {
        console.warn('Could not fetch model info:', infoError);
      }
      
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Cannot connect to backend server. Please make sure the backend is running on http://localhost:5000');
      } else {
        setError(err.message || 'Failed to fetch recommendations');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchRecommendations();
  };

  const getItemTitle = (item) => {
    return item.title || item.name || `Item ${item.item_id}`;
  };

  const getItemDescription = (item) => {
    if (item.genres) {
      return item.genres;
    }
    if (item.category) {
      return item.category;
    }
    if (item.author) {
      return `by ${item.author}`;
    }
    return '';
  };

  const formatRating = (rating) => {
    return rating.toFixed(2);
  };

  if (loading) {
    return (
      <div className="recommendation-results">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Finding the best recommendations for you...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="recommendation-results">
        <div className="error-container">
          <h3>❌ Error Loading Recommendations</h3>
          <p>{error}</p>
          <button onClick={handleRefresh} className="btn btn-retry">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (recommendations.length === 0) {
    return (
      <div className="recommendation-results">
        <div className="no-results">
          <h3>🎯 No Recommendations Found</h3>
          <p>Try selecting different categories or check back later.</p>
          <button onClick={handleRefresh} className="btn btn-primary">
            Refresh
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="recommendation-results">
      <div className="results-header">
        <h3>🎯 Your Personalized Recommendations</h3>
        <div className="results-meta">
          <span className="dataset-badge">{dataset}</span>
          <span className="model-badge">{model.toUpperCase()}</span>
          <span className="count-badge">{recommendations.length} items</span>
        </div>
        <button onClick={handleRefresh} className="btn btn-refresh">
          🔄 Refresh
        </button>
      </div>

      {modelInfo && (
        <div className="model-info">
          <h4>Model Information</h4>
          <div className="model-stats">
            <span>Users: {modelInfo.n_users?.toLocaleString()}</span>
            <span>Items: {modelInfo.n_items?.toLocaleString()}</span>
            {modelInfo.n_factors && <span>Factors: {modelInfo.n_factors}</span>}
          </div>
        </div>
      )}

      <div className="recommendations-grid">
        {recommendations.map((rec, index) => (
          <div key={rec.item_id} className="recommendation-card">
            <div className="card-header">
              <div className="rank">#{index + 1}</div>
              <div className="rating">
                ⭐ {formatRating(rec.predicted_rating)}
              </div>
            </div>
            
            <div className="card-content">
              <h4 className="item-title">{getItemTitle(rec.item_info)}</h4>
              <p className="item-description">{getItemDescription(rec.item_info)}</p>
              
              {rec.item_info.year && (
                <span className="item-year">({rec.item_info.year})</span>
              )}
            </div>
            
            <div className="card-footer">
              <div className="confidence">
                Confidence: {(rec.predicted_rating / 5 * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="results-footer">
        <p className="disclaimer">
          * Recommendations are based on collaborative filtering and may not reflect your exact preferences.
        </p>
      </div>
    </div>
  );
};

export default RecommendationResults;

