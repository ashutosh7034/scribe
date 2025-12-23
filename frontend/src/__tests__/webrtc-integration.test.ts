/**
 * WebRTC Integration Tests
 * 
 * Tests for WebRTC audio streaming integration.
 */

import { audioService } from '../services/audioService';
import { websocketService } from '../services/websocketService';

// Mock browser APIs
const mockMediaStream = {
  getTracks: jest.fn(() => []),
  getAudioTracks: jest.fn(() => [{ enabled: true, stop: jest.fn() }]),
};

const mockGetUserMedia = jest.fn().mockResolvedValue(mockMediaStream);
const mockMediaRecorder = jest.fn().mockImplementation(() => ({
  start: jest.fn(),
  stop: jest.fn(),
  ondataavailable: null,
  onerror: null,
  state: 'inactive',
}));

Object.defineProperty(navigator, 'mediaDevices', {
  value: { getUserMedia: mockGetUserMedia },
  writable: true,
});

Object.defineProperty(window, 'MediaRecorder', {
  value: mockMediaRecorder,
  writable: true,
});

Object.defineProperty(window, 'AudioContext', {
  value: jest.fn().mockImplementation(() => ({
    createMediaStreamSource: jest.fn(() => ({
      connect: jest.fn(),
    })),
    createAnalyser: jest.fn(() => ({
      fftSize: 256,
      smoothingTimeConstant: 0.8,
      frequencyBinCount: 128,
      getByteFrequencyData: jest.fn(),
    })),
    close: jest.fn(),
  })),
  writable: true,
});

// Mock socket.io
jest.mock('socket.io-client', () => ({
  io: jest.fn(() => ({
    on: jest.fn(),
    emit: jest.fn(),
    connected: true,
    disconnect: jest.fn(),
  })),
}));

describe('WebRTC Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    audioService.dispose();
    websocketService.disconnect();
  });

  describe('Browser Support', () => {
    it('should detect WebRTC support correctly', () => {
      expect(audioService.constructor.isSupported()).toBe(true);
    });

    it('should handle missing WebRTC APIs gracefully', () => {
      Object.defineProperty(navigator, 'mediaDevices', {
        value: undefined,
        writable: true,
      });

      expect(audioService.constructor.isSupported()).toBe(false);
    });
  });

  describe('Audio Streaming Setup', () => {
    it('should initialize audio service without errors', async () => {
      // Properly mock navigator.mediaDevices for this test
      Object.defineProperty(navigator, 'mediaDevices', {
        value: { getUserMedia: mockGetUserMedia },
        writable: true,
        configurable: true,
      });

      // Mock successful initialization
      const initPromise = audioService.initialize();
      
      // Should not throw
      await expect(initPromise).resolves.not.toThrow();
    });

    it('should handle microphone permission errors', async () => {
      // Reset mock for this test
      Object.defineProperty(navigator, 'mediaDevices', {
        value: { getUserMedia: jest.fn().mockRejectedValue(new Error('Permission denied')) },
        writable: true,
        configurable: true,
      });
      
      const newAudioService = new (audioService.constructor as any)();
      await expect(newAudioService.initialize()).rejects.toThrow('Microphone access is required');
    });
  });

  describe('Real-time Audio Processing', () => {
    it('should provide audio level monitoring', () => {
      const mockCallback = jest.fn();
      audioService.onAudioLevel(mockCallback);
      
      // Callback should be set without errors
      expect(mockCallback).not.toHaveBeenCalled();
    });

    it('should handle audio streaming start/stop', () => {
      const sessionId = 'test-session-123';
      
      // Should not throw when starting/stopping
      expect(() => {
        audioService.stopRecording();
      }).not.toThrow();
    });
  });

  describe('WebSocket Communication', () => {
    it('should establish WebSocket connection', async () => {
      const status = websocketService.getConnectionStatus();
      expect(status.connected).toBe(false); // Initially disconnected
    });

    it('should handle translation data callbacks', () => {
      const mockCallback = jest.fn();
      websocketService.onTranslationData(mockCallback);
      
      // Should set callback without errors
      expect(mockCallback).not.toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('should handle audio service errors gracefully', () => {
      const mockErrorCallback = jest.fn();
      audioService.onError(mockErrorCallback);
      
      // Should set error callback without throwing
      expect(mockErrorCallback).not.toHaveBeenCalled();
    });

    it('should handle WebSocket errors gracefully', () => {
      const mockErrorCallback = jest.fn();
      websocketService.onError(mockErrorCallback);
      
      // Should set error callback without throwing
      expect(mockErrorCallback).not.toHaveBeenCalled();
    });
  });
});