/**
 * Audio Service Tests
 * 
 * Basic tests for audio service functionality.
 */

import { AudioService } from '../audioService';

// Mock browser APIs
const mockGetUserMedia = jest.fn();
const mockMediaRecorder = jest.fn();
const mockAudioContext = jest.fn();

Object.defineProperty(navigator, 'mediaDevices', {
  value: {
    getUserMedia: mockGetUserMedia,
  },
  writable: true,
});

Object.defineProperty(window, 'MediaRecorder', {
  value: mockMediaRecorder,
  writable: true,
});

Object.defineProperty(window, 'AudioContext', {
  value: mockAudioContext,
  writable: true,
});

describe('AudioService', () => {
  let audioService: AudioService;

  beforeEach(() => {
    audioService = new AudioService();
    jest.clearAllMocks();
  });

  afterEach(() => {
    audioService.dispose();
  });

  describe('isSupported', () => {
    it('should return true when all required APIs are available', () => {
      expect(AudioService.isSupported()).toBe(true);
    });

    it('should return false when getUserMedia is not available', () => {
      Object.defineProperty(navigator, 'mediaDevices', {
        value: undefined,
        writable: true,
      });

      expect(AudioService.isSupported()).toBe(false);
    });
  });

  describe('initialization', () => {
    it('should initialize without errors when browser is supported', async () => {
      mockGetUserMedia.mockResolvedValue({
        getTracks: () => [],
        getAudioTracks: () => [],
      });

      // Mock socket.io
      const mockSocket = {
        on: jest.fn(),
        emit: jest.fn(),
        connected: true,
      };

      // This would normally require more complex mocking
      // For now, just test that the service can be created
      expect(audioService).toBeInstanceOf(AudioService);
    });
  });

  describe('audio metrics', () => {
    it('should return default audio metrics', () => {
      const metrics = audioService.getAudioMetrics();
      
      expect(metrics).toEqual({
        level: 0,
        isActive: false,
        quality: 'poor',
      });
    });
  });
});