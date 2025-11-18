/**
 * API service for communicating with the Recommender System backend
 */

import { config } from '../config';

const API_BASE_URL = config.API_BASE_URL;

class RecommenderAPI {
  /**
   * Check API health status
   */
  async getHealthStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  /**
   * Get available datasets
   */
  async getDatasets() {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`);
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
      throw error;
    }
  }

  /**
   * Get categories/genres for a dataset
   */
  async getCategories(dataset) {
    try {
      const response = await fetch(`${API_BASE_URL}/categories/${dataset}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to fetch categories for ${dataset}:`, error);
      throw error;
    }
  }

  /**
   * Search items in a dataset
   */
  async searchItems(dataset, query, limit = 20) {
    try {
      const params = new URLSearchParams({
        q: query,
        limit: limit.toString()
      });
      const response = await fetch(`${API_BASE_URL}/search/${dataset}?${params}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to search items in ${dataset}:`, error);
      throw error;
    }
  }

  /**
   * Get recommendations for a user
   */
  async getRecommendations(dataset, model, userId, nRecommendations = 10) {
    try {
      const params = new URLSearchParams({
        user_id: userId.toString(),
        n: nRecommendations.toString()
      });
      const response = await fetch(`${API_BASE_URL}/recommend/${dataset}/${model}?${params}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to get recommendations:`, error);
      throw error;
    }
  }

  /**
   * Get recommendations filtered by categories
   */
  async getRecommendationsByCategories(dataset, model, userId, categories, nRecommendations = 10) {
    try {
      const params = new URLSearchParams({
        user_id: userId.toString(),
        n: nRecommendations.toString()
      });
      
      // Add categories as multiple parameters
      categories.forEach(category => {
        params.append('categories', category);
      });
      
      const response = await fetch(`${API_BASE_URL}/recommend-by-categories/${dataset}/${model}?${params}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to get recommendations by categories:`, error);
      throw error;
    }
  }

  /**
   * Predict rating for a user-item pair
   */
  async predictRating(dataset, model, userId, itemId) {
    try {
      const params = new URLSearchParams({
        user_id: userId.toString(),
        item_id: itemId.toString()
      });
      const response = await fetch(`${API_BASE_URL}/predict/${dataset}/${model}?${params}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to predict rating:`, error);
      throw error;
    }
  }

  /**
   * Get model information
   */
  async getModelInfo(dataset, model) {
    try {
      const response = await fetch(`${API_BASE_URL}/model-info/${dataset}/${model}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to get model info:`, error);
      throw error;
    }
  }

  /**
   * Get items from a dataset
   */
  async getItems(dataset) {
    try {
      const response = await fetch(`${API_BASE_URL}/items/${dataset}`);
      return await response.json();
    } catch (error) {
      console.error(`Failed to get items from ${dataset}:`, error);
      throw error;
    }
  }
}

// Create and export a singleton instance
const api = new RecommenderAPI();
export default api;
