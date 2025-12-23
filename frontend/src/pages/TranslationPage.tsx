/**
 * Translation Page Component
 * 
 * Main page for real-time sign language translation interface.
 */

import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  Skeleton,
} from '@mui/material';

import { useAppStore } from '@/store/appStore';
import AvatarViewport from '@/components/avatar/AvatarViewport';
import ControlPanel from '@/components/translation/ControlPanel';
import TranscriptDisplay from '@/components/translation/TranscriptDisplay';
import ConnectionStatus from '@/components/common/ConnectionStatus';

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
    setError,
    resetError,
  } = useAppStore();

  const [transcript, setTranscript] = useState<string>('');
  const [poseSequence, setPoseSequence] = useState<any[]>([]);

  // Handle translation start/stop
  const handleStartTranslation = async () => {
    try {
      resetError();
      setTranslating(true);
      setMicrophoneActive(true);
      
      // TODO: Initialize WebSocket connection and audio capture
      console.log('Starting translation...');
      
    } catch (error) {
      console.error('Failed to start translation:', error);
      setError('Failed to start translation. Please check your microphone permissions.');
      setTranslating(false);
      setMicrophoneActive(false);
    }
  };

  const handleStopTranslation = () => {
    setTranslating(false);
    setMicrophoneActive(false);
    
    // TODO: Close WebSocket connection and stop audio capture
    console.log('Stopping translation...');
  };

  const handleToggleMicrophone = () => {
    if (isTranslating) {
      setMicrophoneActive(!isMicrophoneActive);
    }
  };

  const handleOpenSettings = () => {
    // Navigation handled by parent component
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isTranslating) {
        handleStopTranslation();
      }
    };
  }, []);

  return (
    <Box sx={{ height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column' }}>
      {/* Connection Status */}
      <ConnectionStatus 
        isConnected={isConnected}
        quality={connectionQuality}
        sx={{ mb: 2 }}
      />

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="error" 
          onClose={resetError}
          sx={{ mb: 2 }}
        >
          {error}
        </Alert>
      )}

      {/* Main Content Grid */}
      <Grid container spacing={2} sx={{ flexGrow: 1, minHeight: 0 }}>
        {/* Control Panel */}
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
            sx={{ 
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              minHeight: '400px',
            }}
          >
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
              <Typography variant="h6">
                Sign Language Avatar
              </Typography>
              {selectedAvatar && (
                <Typography variant="body2" color="text.secondary">
                  {selectedAvatar.name}
                </Typography>
              )}
            </Box>

            <Box sx={{ flexGrow: 1, position: 'relative' }}>
              {isLoadingTranslation ? (
                <Box sx={{ p: 2 }}>
                  <Skeleton variant="rectangular" height={300} />
                </Box>
              ) : (
                <AvatarViewport
                  avatar={selectedAvatar}
                  poseSequence={poseSequence}
                  facialExpressions={[]}
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