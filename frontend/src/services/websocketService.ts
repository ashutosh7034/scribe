/**
 * WebSocket Service
 * 
 * Handles real-time communication with the backend for translation streaming.
 */

import { io, Socket } from 'socket.io-client';
import { TranslationFrame, WebSocketMessage, PoseKeyframe, FacialExpressionKeyframe } from '../types';

export interface WebSocketConfig {
  url?: string;
  timeout?: number;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
}

export interface TranslationStreamData {
  sessionId: string;
  transcript: string;
  poseSequence: PoseKeyframe[];
  facialExpressions: FacialExpressionKeyframe[];
  confidence: number;
  processingTime: number;
}

export class WebSocketService {
  private socket: Socket | null = null;
  private config: Required<WebSocketConfig>;
  private isConnected = false;
  private currentSessionId: string | null = null;

  // Event callbacks
  private onConnectedCallback?: (connected: boolean) => void;
  private onTranslationDataCallback?: (data: TranslationStreamData) => void;
  private onTranscriptUpdateCallback?: (transcript: string) => void;
  private onErrorCallback?: (error: Error) => void;

  constructor(config: WebSocketConfig = {}) {
    this.config = {
      url: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
      timeout: 5000,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      ...config,
    };
  }

  /**
   * Initialize WebSocket connection
   */
  async connect(): Promise<void> {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }

    return new Promise((resolve, reject) => {
      this.socket = io(this.config.url, {
        transports: ['websocket'],
        timeout: this.config.timeout,
        reconnectionAttempts: this.config.reconnectionAttempts,
        reconnectionDelay: this.config.reconnectionDelay,
      });

      this.setupEventHandlers();

      this.socket.on('connect', () => {
        console.log('WebSocket connected successfully');
        this.isConnected = true;
        this.onConnectedCallback?.(true);
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection failed:', error);
        this.isConnected = false;
        this.onConnectedCallback?.(false);
        this.onErrorCallback?.(new Error(`Connection failed: ${error.message}`));
        reject(error);
      });

      // Set connection timeout
      setTimeout(() => {
        if (!this.isConnected) {
          reject(new Error('WebSocket connection timeout'));
        }
      }, this.config.timeout);
    });
  }

  /**
   * Set up WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.onConnectedCallback?.(false);
    });

    this.socket.on('reconnect', () => {
      console.log('WebSocket reconnected');
      this.isConnected = true;
      this.onConnectedCallback?.(true);
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.onErrorCallback?.(new Error(error.message || 'WebSocket error'));
    });

    // Translation-specific events
    this.socket.on('translation_frame', (data: TranslationFrame) => {
      this.handleTranslationFrame(data);
    });

    this.socket.on('transcript_update', (data: { session_id: string; transcript: string }) => {
      if (data.session_id === this.currentSessionId) {
        this.onTranscriptUpdateCallback?.(data.transcript);
      }
    });

    this.socket.on('translation_complete', (data: TranslationStreamData) => {
      if (data.sessionId === this.currentSessionId) {
        this.onTranslationDataCallback?.(data);
      }
    });

    this.socket.on('session_error', (data: { session_id: string; error: string }) => {
      if (data.session_id === this.currentSessionId) {
        this.onErrorCallback?.(new Error(data.error));
      }
    });
  }

  /**
   * Handle incoming translation frames
   */
  private handleTranslationFrame(frame: TranslationFrame): void {
    if (frame.session_id !== this.currentSessionId) return;

    // Convert frame data to pose sequence format
    const poseKeyframe: PoseKeyframe = {
      timestamp: frame.timestamp,
      joints: frame.pose_data,
    };

    const facialKeyframe: FacialExpressionKeyframe = {
      timestamp: frame.timestamp,
      expression: frame.facial_expression,
      intensity: 1.0, // Default intensity
    };

    // Emit as translation data
    this.onTranslationDataCallback?.({
      sessionId: frame.session_id,
      transcript: '', // Will be updated separately
      poseSequence: [poseKeyframe],
      facialExpressions: [facialKeyframe],
      confidence: 1.0,
      processingTime: 0,
    });
  }

  /**
   * Start a new translation session
   */
  async startSession(sessionId: string, config?: {
    sourceLanguage?: string;
    targetSignLanguage?: string;
    avatarId?: number;
  }): Promise<void> {
    if (!this.socket?.connected) {
      throw new Error('WebSocket not connected');
    }

    this.currentSessionId = sessionId;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Session start timeout'));
      }, 5000);

      this.socket!.emit('start_translation_session', {
        session_id: sessionId,
        timestamp: Date.now(),
        config: {
          source_language: config?.sourceLanguage || 'en',
          target_sign_language: config?.targetSignLanguage || 'asl',
          avatar_id: config?.avatarId,
        },
      });

      this.socket!.once('session_started', (data: { session_id: string; status: string }) => {
        clearTimeout(timeout);
        if (data.session_id === sessionId && data.status === 'success') {
          console.log('Translation session started:', sessionId);
          resolve();
        } else {
          reject(new Error('Failed to start session'));
        }
      });
    });
  }

  /**
   * End the current translation session
   */
  async endSession(): Promise<void> {
    if (!this.socket?.connected || !this.currentSessionId) {
      return;
    }

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        resolve(); // Don't fail if no response
      }, 2000);

      this.socket!.emit('end_translation_session', {
        session_id: this.currentSessionId,
        timestamp: Date.now(),
      });

      this.socket!.once('session_ended', () => {
        clearTimeout(timeout);
        console.log('Translation session ended:', this.currentSessionId);
        this.currentSessionId = null;
        resolve();
      });
    });
  }

  /**
   * Send audio data to backend
   */
  sendAudioData(audioData: ArrayBuffer): void {
    if (!this.socket?.connected || !this.currentSessionId) {
      console.warn('Cannot send audio data: not connected or no active session');
      return;
    }

    this.socket.emit('audio_data', {
      session_id: this.currentSessionId,
      timestamp: Date.now(),
      audio_data: Array.from(new Uint8Array(audioData)),
    });
  }

  /**
   * Send text for translation (fallback method)
   */
  sendTextForTranslation(text: string): void {
    if (!this.socket?.connected || !this.currentSessionId) {
      console.warn('Cannot send text: not connected or no active session');
      return;
    }

    this.socket.emit('translate_text', {
      session_id: this.currentSessionId,
      timestamp: Date.now(),
      text: text,
    });
  }

  /**
   * Set callback for connection status changes
   */
  onConnected(callback: (connected: boolean) => void): void {
    this.onConnectedCallback = callback;
  }

  /**
   * Set callback for translation data updates
   */
  onTranslationData(callback: (data: TranslationStreamData) => void): void {
    this.onTranslationDataCallback = callback;
  }

  /**
   * Set callback for transcript updates
   */
  onTranscriptUpdate(callback: (transcript: string) => void): void {
    this.onTranscriptUpdateCallback = callback;
  }

  /**
   * Set callback for errors
   */
  onError(callback: (error: Error) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): {
    connected: boolean;
    sessionId: string | null;
    quality: 'excellent' | 'good' | 'fair' | 'poor';
  } {
    return {
      connected: this.isConnected,
      sessionId: this.currentSessionId,
      quality: this.isConnected ? 'excellent' : 'poor',
    };
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    if (this.currentSessionId) {
      this.endSession();
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.isConnected = false;
    this.currentSessionId = null;
  }
}

// Export singleton instance
export const websocketService = new WebSocketService();
export default websocketService;