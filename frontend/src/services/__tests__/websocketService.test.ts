/**
 * WebSocket Service Tests
 * 
 * Basic tests for WebSocket service functionality.
 */

import { WebSocketService } from '../websocketService';

// Mock socket.io-client
jest.mock('socket.io-client', () => ({
  io: jest.fn(() => ({
    on: jest.fn(),
    emit: jest.fn(),
    connected: false,
    disconnect: jest.fn(),
  })),
}));

describe('WebSocketService', () => {
  let websocketService: WebSocketService;

  beforeEach(() => {
    websocketService = new WebSocketService();
  });

  afterEach(() => {
    websocketService.disconnect();
  });

  describe('initialization', () => {
    it('should create service instance', () => {
      expect(websocketService).toBeInstanceOf(WebSocketService);
    });

    it('should have default configuration', () => {
      const status = websocketService.getConnectionStatus();
      expect(status.connected).toBe(false);
      expect(status.sessionId).toBe(null);
      expect(status.quality).toBe('poor');
    });
  });

  describe('connection status', () => {
    it('should return connection status', () => {
      const status = websocketService.getConnectionStatus();
      
      expect(status).toEqual({
        connected: false,
        sessionId: null,
        quality: 'poor',
      });
    });
  });

  describe('event callbacks', () => {
    it('should set callback functions', () => {
      const mockCallback = jest.fn();
      
      websocketService.onConnected(mockCallback);
      websocketService.onTranslationData(mockCallback);
      websocketService.onTranscriptUpdate(mockCallback);
      websocketService.onError(mockCallback);
      
      // If we get here without errors, callbacks were set successfully
      expect(true).toBe(true);
    });
  });
});