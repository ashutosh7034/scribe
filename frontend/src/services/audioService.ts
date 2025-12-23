/**
 * Audio Service
 * 
 * Handles WebRTC audio capture, processing, and streaming for real-time translation.
 */

import { io, Socket } from 'socket.io-client';

export interface AudioServiceConfig {
  sampleRate?: number;
  channels?: number;
  bufferSize?: number;
  socketUrl?: string;
}

export interface AudioMetrics {
  level: number;
  isActive: boolean;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
}

export class AudioService {
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private socket: Socket | null = null;
  private isRecording = false;
  private sessionId: string | null = null;
  
  private config: Required<AudioServiceConfig>;
  private onAudioLevelCallback?: (level: number) => void;
  private onErrorCallback?: (error: Error) => void;
  private onConnectionChangeCallback?: (connected: boolean, quality?: string) => void;

  constructor(config: AudioServiceConfig = {}) {
    this.config = {
      sampleRate: 16000,
      channels: 1,
      bufferSize: 4096,
      socketUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
      ...config,
    };
  }

  /**
   * Initialize audio capture and WebSocket connection
   */
  async initialize(): Promise<void> {
    try {
      // Request microphone permissions
      await this.requestMicrophonePermission();
      
      // Initialize WebSocket connection
      await this.initializeWebSocket();
      
      // Set up audio analysis
      this.setupAudioAnalysis();
      
    } catch (error) {
      console.error('Failed to initialize audio service:', error);
      this.onErrorCallback?.(error as Error);
      throw error;
    }
  }

  /**
   * Request microphone permission and set up media stream
   */
  private async requestMicrophonePermission(): Promise<void> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (error) {
      console.error('Microphone permission denied:', error);
      throw new Error('Microphone access is required for translation. Please allow microphone permissions.');
    }
  }

  /**
   * Initialize WebSocket connection for real-time audio streaming
   */
  private async initializeWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket = io(this.config.socketUrl, {
        transports: ['websocket'],
        timeout: 5000,
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.onConnectionChangeCallback?.(true, 'excellent');
        resolve();
      });

      this.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        this.onConnectionChangeCallback?.(false);
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.onConnectionChangeCallback?.(false);
        reject(error);
      });

      this.socket.on('translation_response', (data) => {
        // Handle translation responses from backend
        console.log('Translation response received:', data);
      });

      this.socket.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.onErrorCallback?.(new Error(error.message || 'WebSocket error'));
      });
    });
  }

  /**
   * Set up audio analysis for level monitoring
   */
  private setupAudioAnalysis(): void {
    if (!this.mediaStream) return;

    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.8;
    
    source.connect(this.analyser);
    
    // Start monitoring audio levels
    this.monitorAudioLevel();
  }

  /**
   * Monitor audio levels for UI feedback
   */
  private monitorAudioLevel(): void {
    if (!this.analyser) return;

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    
    const updateLevel = () => {
      if (!this.analyser || !this.isRecording) return;
      
      this.analyser.getByteFrequencyData(dataArray);
      
      // Calculate RMS (Root Mean Square) for audio level
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / dataArray.length);
      const level = Math.min(100, (rms / 128) * 100);
      
      this.onAudioLevelCallback?.(level);
      
      if (this.isRecording) {
        requestAnimationFrame(updateLevel);
      }
    };
    
    updateLevel();
  }

  /**
   * Start audio recording and streaming
   */
  async startRecording(sessionId: string): Promise<void> {
    if (!this.mediaStream || !this.socket) {
      throw new Error('Audio service not initialized');
    }

    if (this.isRecording) {
      console.warn('Recording already in progress');
      return;
    }

    try {
      this.sessionId = sessionId;
      this.isRecording = true;

      // Set up MediaRecorder for audio streaming
      this.mediaRecorder = new MediaRecorder(this.mediaStream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.socket?.connected) {
          // Convert blob to array buffer and send via WebSocket
          event.data.arrayBuffer().then((buffer) => {
            this.socket?.emit('audio_data', {
              session_id: this.sessionId,
              timestamp: Date.now(),
              audio_data: Array.from(new Uint8Array(buffer)),
            });
          });
        }
      };

      this.mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        this.onErrorCallback?.(new Error('Audio recording error'));
      };

      // Start recording with small time slices for real-time streaming
      this.mediaRecorder.start(100); // 100ms chunks

      // Notify backend of session start
      this.socket.emit('session_start', {
        session_id: sessionId,
        timestamp: Date.now(),
        audio_config: {
          sample_rate: this.config.sampleRate,
          channels: this.config.channels,
        },
      });

      console.log('Audio recording started');
      
    } catch (error) {
      this.isRecording = false;
      console.error('Failed to start recording:', error);
      throw error;
    }
  }

  /**
   * Stop audio recording and streaming
   */
  stopRecording(): void {
    if (!this.isRecording) return;

    this.isRecording = false;

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    if (this.socket && this.sessionId) {
      this.socket.emit('session_end', {
        session_id: this.sessionId,
        timestamp: Date.now(),
      });
    }

    this.sessionId = null;
    console.log('Audio recording stopped');
  }

  /**
   * Toggle microphone mute state
   */
  toggleMicrophone(muted: boolean): void {
    if (!this.mediaStream) return;

    this.mediaStream.getAudioTracks().forEach((track) => {
      track.enabled = !muted;
    });
  }

  /**
   * Get current audio metrics
   */
  getAudioMetrics(): AudioMetrics {
    return {
      level: 0, // Will be updated by monitoring
      isActive: this.isRecording,
      quality: this.socket?.connected ? 'excellent' : 'poor',
    };
  }

  /**
   * Set callback for audio level updates
   */
  onAudioLevel(callback: (level: number) => void): void {
    this.onAudioLevelCallback = callback;
  }

  /**
   * Set callback for errors
   */
  onError(callback: (error: Error) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Set callback for connection changes
   */
  onConnectionChange(callback: (connected: boolean, quality?: string) => void): void {
    this.onConnectionChangeCallback = callback;
  }

  /**
   * Check if browser supports required features
   */
  static isSupported(): boolean {
    return !!(
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function' &&
      window.MediaRecorder &&
      (window.AudioContext || (window as any).webkitAudioContext)
    );
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopRecording();

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }

    this.analyser = null;
    this.mediaRecorder = null;
  }
}

// Export singleton instance
export const audioService = new AudioService();
export default audioService;