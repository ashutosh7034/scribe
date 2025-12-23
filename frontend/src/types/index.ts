/**
 * Core Type Definitions
 * 
 * Shared TypeScript interfaces and types for the Scribe application.
 */

// API Response Types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
}

// Translation Types
export interface TranslationRequest {
  text: string;
  source_language?: string;
  target_sign_language?: string;
  emotion_intensity?: number;
  signing_speed?: number;
  avatar_id?: number;
  session_id?: string;
}

export interface TranslationResponse {
  session_id: string;
  original_text: string;
  normalized_text: string;
  pose_sequence: PoseKeyframe[];
  facial_expressions: FacialExpressionKeyframe[];
  duration_ms: number;
  emotion_detected?: string;
  confidence_score: number;
  processing_time_ms: number;
}

export interface PoseKeyframe {
  timestamp: number;
  joints: Record<string, Vector3>;
}

export interface FacialExpressionKeyframe {
  timestamp: number;
  expression: string;
  intensity: number;
}

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

// Avatar Types
export interface Avatar {
  id: number;
  name: string;
  description?: string;
  mesh_url: string;
  texture_url: string;
  skeleton_config: Record<string, any>;
  gender?: string;
  ethnicity?: string;
  age_range?: string;
  is_active: boolean;
  is_premium: boolean;
  created_at: string;
}

export interface AvatarCustomization {
  id: number;
  user_id: number;
  avatar_id: number;
  customization_name: string;
  skin_tone?: string;
  hair_style?: string;
  hair_color?: string;
  eye_color?: string;
  clothing_style: string;
  clothing_color?: string;
  accessories?: string[];
  signing_space: 'compact' | 'standard' | 'expansive';
  formality_level: 'casual' | 'formal';
  created_at: string;
  updated_at: string;
  avatar: Avatar;
}

// User Types
export interface User {
  id: number;
  email?: string;
  username?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserPreferences {
  id: number;
  user_id: number;
  preferred_sign_language: string;
  signing_speed: number;
  emotion_intensity: number;
  selected_avatar_id?: number;
  high_contrast_mode: boolean;
  font_scaling: number;
  preferred_quality: 'low' | 'standard' | 'high';
  created_at: string;
  updated_at: string;
}

// Translation Session Types
export interface TranslationSession {
  id: number;
  session_id: string;
  user_id?: number;
  start_time: string;
  end_time?: string;
  duration_seconds?: number;
  source_language: string;
  target_sign_language: string;
  avatar_id: number;
  total_words_translated: number;
  average_latency_ms?: number;
  error_count: number;
  user_satisfaction_score?: number;
  accuracy_feedback_count: number;
  client_platform?: string;
  client_version?: string;
}

// Feedback Types
export interface TranslationFeedback {
  id: number;
  session_id: number;
  original_text: string;
  feedback_type: 'accuracy' | 'speed' | 'expression' | 'other';
  rating?: number;
  comment?: string;
  suggested_correction?: string;
  is_emergency_phrase: boolean;
  timestamp_in_session?: number;
  emotion_detected?: string;
  confidence_score?: number;
  is_processed: boolean;
  processing_priority: number;
  created_at: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'translation_frame' | 'audio_data' | 'session_start' | 'session_end' | 'error';
  session_id: string;
  timestamp: number;
  data?: any;
}

export interface TranslationFrame {
  type: 'translation_frame';
  session_id: string;
  timestamp: number;
  pose_data: Record<string, Vector3>;
  facial_expression: string;
}

// UI State Types
export interface AppState {
  // User preferences
  highContrastMode: boolean;
  fontScaling: 'small' | 'normal' | 'large' | 'extra-large';
  
  // Translation state
  isTranslating: boolean;
  currentSession?: TranslationSession;
  selectedAvatar?: Avatar;
  
  // Audio state
  isMicrophoneActive: boolean;
  audioLevel: number;
  
  // Connection state
  isConnected: boolean;
  connectionQuality: 'excellent' | 'good' | 'fair' | 'poor';
  
  // Error state
  error?: string;
  
  // Loading states
  isLoadingAvatars: boolean;
  isLoadingTranslation: boolean;
}

// Component Props Types
export interface AvatarViewportProps {
  avatar?: Avatar;
  poseSequence?: PoseKeyframe[];
  facialExpressions?: FacialExpressionKeyframe[];
  isPlaying: boolean;
  quality: 'low' | 'standard' | 'high';
}

export interface ControlPanelProps {
  isTranslating: boolean;
  isMicrophoneActive: boolean;
  audioLevel: number;
  onStartTranslation: () => void;
  onStopTranslation: () => void;
  onToggleMicrophone: () => void;
  onOpenSettings: () => void;
}

// API Client Types
export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
}

// Error Types
export interface ScribeError extends Error {
  code?: string;
  statusCode?: number;
  details?: any;
}

// Accessibility Types
export interface AccessibilityOptions {
  highContrast: boolean;
  fontSize: number;
  keyboardNavigation: boolean;
  screenReaderSupport: boolean;
  reducedMotion: boolean;
}