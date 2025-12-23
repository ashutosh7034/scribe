/**
 * Control Panel Component
 * 
 * Main controls for translation functionality.
 */

import React from 'react';
import {
  Box,
  Button,
  IconButton,
  LinearProgress,
  Typography,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Mic as MicIcon,
  MicOff as MicOffIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  VolumeUp as VolumeIcon,
} from '@mui/icons-material';
import { ControlPanelProps } from '../../types';

const ControlPanel: React.FC<ControlPanelProps> = ({
  isTranslating,
  isMicrophoneActive,
  audioLevel,
  onStartTranslation,
  onStopTranslation,
  onToggleMicrophone,
  onOpenSettings,
}) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
      {/* Main Translation Control */}
      <Box sx={{ textAlign: 'center' }}>
        {!isTranslating ? (
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayIcon />}
            onClick={onStartTranslation}
            sx={{ 
              minHeight: '56px',
              width: '100%',
              fontSize: '1.1rem',
            }}
          >
            Start Translation
          </Button>
        ) : (
          <Button
            variant="contained"
            color="error"
            size="large"
            startIcon={<StopIcon />}
            onClick={onStopTranslation}
            sx={{ 
              minHeight: '56px',
              width: '100%',
              fontSize: '1.1rem',
            }}
          >
            Stop Translation
          </Button>
        )}
      </Box>

      <Divider />

      {/* Microphone Controls */}
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Microphone
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Tooltip title={isMicrophoneActive ? 'Mute microphone' : 'Unmute microphone'}>
            <IconButton
              onClick={onToggleMicrophone}
              disabled={!isTranslating}
              color={isMicrophoneActive ? 'primary' : 'default'}
              sx={{ minHeight: '44px', minWidth: '44px' }}
            >
              {isMicrophoneActive ? <MicIcon /> : <MicOffIcon />}
            </IconButton>
          </Tooltip>
          
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Audio Level
            </Typography>
            <LinearProgress
              variant="determinate"
              value={audioLevel}
              sx={{ 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: audioLevel > 80 ? 'error.main' : 
                                   audioLevel > 50 ? 'warning.main' : 'success.main',
                },
              }}
            />
          </Box>
        </Box>

        <Typography variant="caption" color="text.secondary">
          {isMicrophoneActive ? 'Microphone is active' : 'Microphone is muted'}
        </Typography>
      </Box>

      <Divider />

      {/* Translation Status */}
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Status
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: isTranslating ? 'success.main' : 'grey.400',
            }}
          />
          <Typography variant="body2">
            {isTranslating ? 'Translating' : 'Ready'}
          </Typography>
        </Box>

        {isTranslating && (
          <Typography variant="caption" color="text.secondary">
            Real-time translation active
          </Typography>
        )}
      </Box>

      {/* Spacer */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Settings Button */}
      <Box sx={{ textAlign: 'center' }}>
        <Tooltip title="Open settings">
          <IconButton
            onClick={onOpenSettings}
            sx={{ 
              minHeight: '44px', 
              minWidth: '44px',
              border: 1,
              borderColor: 'divider',
            }}
          >
            <SettingsIcon />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default ControlPanel;