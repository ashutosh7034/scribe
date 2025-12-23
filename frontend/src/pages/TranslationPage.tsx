/**
 * Translation Page Component
 * 
 * Main page for real-time sign language translation interface.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  Skeleton,
} from '@mui/material';

import { useAppStore } from '../store/appStore';
import AvatarViewport from '../components/avatar/AvatarViewport';
import ControlPanel from '../components/translation/ControlPanel';
import TranscriptDisplay from '../components/translation/TranscriptDisplay';
import ConnectionStatus from '../components/common/ConnectionStatus';
import BrowserCompatibility from '../components/common/BrowserCompatibility';
import { audioService, AudioService } from '../services/audioService';
import { websocketService } from '../services/websocketService';
import { apiClient } from '../services/apiClient';
import { PoseKeyframe, FacialExpressionKeyframe } from '../types';

const TranslationPage: React.FC = () => {
  const {
    selectedAvatar,
    isTranslating,
    isMicrophoneActive,
    audioLevel,
    isConnected,
    connectionQuality,
    error,
    isLoadingTranslation,
    setTranslating,
    setMicrophoneActive,
    setAudioLevel,
    setConnectionState,
    setError,
    resetError,
    setLoadingStates,
  } = useAppStore();

  const [transcript, setTranscript] = useState<string>('');
  const [poseSequence, setPoseSequence] = useState<PoseKeyframe[]>([]);
  const [facialExpressions, setFacialExpressions] = useState<FacialExpressionKeyframe[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  
  const isInitialized = useRef(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Initialize services on component mount
  useEffect(() => {
    const initializeServices = async () => {
      if (isInitialized.current) return;
      
      try {
        // Check browser support
        if (!AudioService.isSupported()) {
          setError('Your browser does not support the required audio features. Please use a modern browser.');
          return;
        }

        setLoadingStates({ isLoadingTranslation: true });

        // Set up service callbacks
        audioService.onAudioLevel((level) => {
          setAudioLevel(level);
        });

        audioService.onError((error) => {
          console.error('Audio service error:', error);
          setError(error.message);
          setTranslating(false);
          setMicrophoneActive(false);
        });

        audioService.onConnectionChange((connected, quality) => {
          setConnectionState(connected, quality as any);
        });

        websocketService.onConnected((connected) => {
          setConnectionState(connected, connected ? 'excellent' : 'poor');
        });

        websocketService.onTranslationData((data) => {
          setPoseSequence(prev => [...prev, ...data.poseSequence]);
          setFacialExpressions(prev => [...prev, ...data.facialExpressions]);
        });

        websocketService.onTranscriptUpdate((newTranscript) => {
          setTranscript(newTranscript);
        });

        websocketService.onError((error) => {
          console.error('WebSocket service error:', error);
          setError(error.message);
        });

        // Initialize services
        await websocketService.connect();
        await audioService.initialize();

        isInitialized.current = true;
        setLoadingStates({ isLoadingTranslation: false });

      } catch (error) {
        console.error('Failed to initialize services:', error);
        setError(error instanceof Error ? error.message : 'Failed to initialize audio services');
        setLoadingStates({ isLoadingTranslation: false });
      }
    };

    initializeServices();

    // Cleanup function
    cleanupRef.current = () => {
      if (isTranslating) {
        handleStopTranslation();
      }
      audioService.dispose();
      websocketService.disconnect();
    };

    return cleanupRef.current;
  }, []);

  // Handle translation start/stop
  const handleStartTranslation = async () => {
    try {
      resetError();
      
      if (!selectedAvatar) {
        setError('Please select an avatar before starting translation');
        return;
      }

      setLoadingStates({ isLoadingTranslation: true });

      // Create a new translation session
      const sessionData = await apiClient.createTranslationSession({
        avatar_id: selectedAvatar.id,
        source_language: 'en',
        target_sign_language: 'asl',
        client_platform: 'web',
        client_version: '1.0.0',
      });

      const sessionId = sessionData.session_id || `session_${Date.now()}`;
      setCurrentSessionId(sessionId);

      // Start WebSocket session
      await websocketService.startSession(sessionId, {
        sourceLanguage: 'en',
        targetSignLanguage: 'asl',
        avatarId: selectedAvatar.id,
      });

      // Start audio recording
      await audioService.startRecording(sessionId);

      setTranslating(true);
      setMicrophoneActive(true);
      setLoadingStates({ isLoadingTranslation: false });
      
      console.log('Translation started with session:', sessionId);
      
    } catch (error) {
      console.error('Failed to start translation:', error);
      setError(error instanceof Error ? error.message : 'Failed to start translation. Please check your microphone permissions.');
      setTranslating(false);
      setMicrophoneActive(false);
      setLoadingStates({ isLoadingTranslation: false });
      setCurrentSessionId(null);
    }
  };

  const handleStopTranslation = async () => {
    try {
      setTranslating(false);
      setMicrophoneActive(false);
      
      // Stop audio recording
      audioService.stopRecording();
      
      // End WebSocket session
      if (currentSessionId) {
        await websocketService.endSession();
      }
      
      setCurrentSessionId(null);
      console.log('Translation stopped');
      
    } catch (error) {
      console.error('Error stopping translation:', error);
      // Don't show error to user for stop operation
    }
  };

  const handleToggleMicrophone = () => {
    if (isTranslating) {
      const newMutedState = isMicrophoneActive;
      setMicrophoneActive(!isMicrophoneActive);
      audioService.toggleMicrophone(newMutedState);
    }
  };

  const handleOpenSettings = () => {
    // Navigation handled by parent component
  };

  // Clear pose sequence when translation stops
  useEffect(() => {
    if (!isTranslating) {
      setPoseSequence([]);
      setFacialExpressions([]);
      setTranscript('');
    }
  }, [isTranslating]);

  return (
    <Box style={{ height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      {/* Browser Compatibility Check */}
      <BrowserCompatibility />

      {/* Connection Status */}
      <ConnectionStatus 
        isConnected={isConnected}
        quality={connectionQuality}
        style={{ marginBottom: 16 }}
      />

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="error" 
          onClose={resetError}
          style={{ marginBottom: 16 }}
        >
          {error}
        </Alert>
      )}

      {/* Main Content Grid */}
      <Grid container spacing={2} style={{ flexGrow: 1, minHeight: 0 }}>
        {/* Control Panel */}
        <Grid item xs={12} md={3}>
          <Paper 
            elevation={2} 
            style={{ 
              padding: 16, 
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <Typography variant="h6" gutterBottom>
              Controls
            </Typography>
            
            <ControlPanel
              isTranslating={isTranslating}
              isMicrophoneActive={isMicrophoneActive}
              audioLevel={audioLevel}
              onStartTranslation={handleStartTranslation}
              onStopTranslation={handleStopTranslation}
              onToggleMicrophone={handleToggleMicrophone}
              onOpenSettings={handleOpenSettings}
            />
          </Paper>
        </Grid>

        {/* Avatar Viewport */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={2} 
            style={{ 
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              minHeight: '400px',
            }}
          >
            <Box style={{ padding: 16, borderBottom: '1px solid #e0e0e0' }}>
              <Typography variant="h6">
                Sign Language Avatar
              </Typography>
              {selectedAvatar && (
                <Typography variant="body2" color="text.secondary">
                  {selectedAvatar.name}
                </Typography>
              )}
            </Box>

            <Box style={{ flexGrow: 1, position: 'relative' }}>
              {isLoadingTranslation ? (
                <Box style={{ padding: 16 }}>
                  <Skeleton variant="rectangular" height={300} />
                  <Typography variant="body2" style={{ marginTop: 8, textAlign: 'center' }}>
                    Initializing translation services...
                  </Typography>
                </Box>
              ) : (
                <AvatarViewport
                  avatar={selectedAvatar}
                  poseSequence={poseSequence}
                  facialExpressions={facialExpressions}
                  isPlaying={isTranslating}
                  quality="standard"
                />
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Chat/Transcript Panel */}
        <Grid item xs={12} md={3}>
          <Paper 
            elevation={2} 
            sx={{ 
              p: 2, 
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <Typography variant="h6" gutterBottom>
              Live Transcript
            </Typography>
            
            <TranscriptDisplay
              transcript={transcript}
              isActive={isTranslating}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TranslationPage;