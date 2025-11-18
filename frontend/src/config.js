// Configuration for the frontend application
export const config = {
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  DEFAULT_USER_ID: 1,
  DEFAULT_MODEL: 'svd',
  MAX_RECOMMENDATIONS: 10
};

