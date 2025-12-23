/**
 * API Client Service
 * 
 * Centralized HTTP client for communicating with the Scribe backend API.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { 
  ApiResponse, 
  TranslationRequest, 
  TranslationResponse,
  Avatar,
  AvatarCustomization,
  User,
  UserPreferences,
  TranslationSession,
  TranslationFeedback,
  ScribeError 
} from '@/types';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add authentication token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        const scribeError: ScribeError = new Error(
          error.response?.data?.message || error.message || 'An error occurred'
        );
        scribeError.code = error.code;
        scribeError.statusCode = error.response?.status;
        scribeError.details = error.response?.data;
        
        return Promise.reject(scribeError);
      }
    );
  }

  // Health endpoints
  async healthCheck(): Promise<ApiResponse> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async detailedHealthCheck(): Promise<ApiResponse> {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }

  // Translation endpoints
  async translateText(request: TranslationRequest): Promise<TranslationResponse> {
    const response = await this.client.post('/translate', request);
    return response.data;
  }

  async createTranslationSession(sessionData: {
    user_id?: number;
    source_language?: string;
    target_sign_language?: string;
    avatar_id: number;
    client_platform?: string;
    client_version?: string;
  }): Promise<TranslationSession> {
    const response = await this.client.post('/translate/sessions', sessionData);
    return response.data;
  }

  async getTranslationSession(sessionId: string): Promise<TranslationSession> {
    const response = await this.client.get(`/translate/sessions/${sessionId}`);
    return response.data;
  }

  async submitFeedback(feedback: {
    session_id: number;
    original_text: string;
    feedback_type: 'accuracy' | 'speed' | 'expression' | 'other';
    rating?: number;
    comment?: string;
    suggested_correction?: string;
    is_emergency_phrase?: boolean;
    timestamp_in_session?: number;
  }): Promise<TranslationFeedback> {
    const response = await this.client.post('/translate/feedback', feedback);
    return response.data;
  }

  async getSessionFeedback(sessionId: number): Promise<TranslationFeedback[]> {
    const response = await this.client.get(`/translate/feedback/${sessionId}`);
    return response.data;
  }

  // Avatar endpoints
  async getAvatars(params?: {
    page?: number;
    page_size?: number;
    gender?: string;
    ethnicity?: string;
    is_premium?: boolean;
  }): Promise<{
    avatars: Avatar[];
    total: number;
    page: number;
    page_size: number;
  }> {
    const response = await this.client.get('/avatars', { params });
    return response.data;
  }

  async getAvatar(avatarId: number): Promise<Avatar> {
    const response = await this.client.get(`/avatars/${avatarId}`);
    return response.data;
  }

  async createAvatarCustomization(customization: {
    avatar_id: number;
    customization_name?: string;
    skin_tone?: string;
    hair_style?: string;
    hair_color?: string;
    eye_color?: string;
    clothing_style?: string;
    clothing_color?: string;
    accessories?: string[];
    signing_space?: 'compact' | 'standard' | 'expansive';
    formality_level?: 'casual' | 'formal';
  }, userId: number): Promise<AvatarCustomization> {
    const response = await this.client.post('/avatars/customizations', customization, {
      params: { user_id: userId }
    });
    return response.data;
  }

  async getUserCustomizations(userId: number): Promise<AvatarCustomization[]> {
    const response = await this.client.get(`/avatars/customizations/${userId}`);
    return response.data;
  }

  async updateAvatarCustomization(
    customizationId: number, 
    updates: Partial<AvatarCustomization>,
    userId: number
  ): Promise<AvatarCustomization> {
    const response = await this.client.put(
      `/avatars/customizations/${customizationId}`, 
      updates,
      { params: { user_id: userId } }
    );
    return response.data;
  }

  async deleteAvatarCustomization(customizationId: number, userId: number): Promise<void> {
    await this.client.delete(`/avatars/customizations/${customizationId}`, {
      params: { user_id: userId }
    });
  }

  // User endpoints
  async createUser(userData: {
    email?: string;
    username?: string;
  }): Promise<User> {
    const response = await this.client.post('/users', userData);
    return response.data;
  }

  async getUser(userId: number): Promise<User> {
    const response = await this.client.get(`/users/${userId}`);
    return response.data;
  }

  async getUserPreferences(userId: number): Promise<UserPreferences> {
    const response = await this.client.get(`/users/${userId}/preferences`);
    return response.data;
  }

  async updateUserPreferences(
    userId: number, 
    preferences: Partial<UserPreferences>
  ): Promise<UserPreferences> {
    const response = await this.client.put(`/users/${userId}/preferences`, preferences);
    return response.data;
  }

  async deleteUser(userId: number): Promise<void> {
    await this.client.delete(`/users/${userId}`);
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;