/**
 * Transcript Display Component
 * 
 * Shows live transcript of spoken text being translated.
 */

import React, { useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
} from '@mui/material';

interface TranscriptDisplayProps {
  transcript: string;
  isActive: boolean;
}

const TranscriptDisplay: React.FC<TranscriptDisplayProps> = ({
  transcript,
  isActive,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new content is added
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Status Indicator */}
      <Box sx={{ mb: 2 }}>
        <Chip
          label={isActive ? 'Live' : 'Inactive'}
          color={isActive ? 'success' : 'default'}
          size="small"
          variant={isActive ? 'filled' : 'outlined'}
        />
      </Box>

      {/* Transcript Content */}
      <Paper
        variant="outlined"
        sx={{
          flexGrow: 1,
          p: 2,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Box
          ref={scrollRef}
          sx={{
            flexGrow: 1,
            overflowY: 'auto',
            maxHeight: '100%',
          }}
        >
          {transcript ? (
            <Typography
              variant="body1"
              sx={{
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {transcript}
            </Typography>
          ) : (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ fontStyle: 'italic' }}
            >
              {isActive 
                ? 'Listening for speech...' 
                : 'Start translation to see live transcript'
              }
            </Typography>
          )}
        </Box>

        {/* Typing Indicator */}
        {isActive && (
          <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: 'primary.main',
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': {
                  '0%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                  '100%': { opacity: 1 },
                },
              }}
            />
            <Typography variant="caption" color="text.secondary">
              Processing...
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Word Count */}
      {transcript && (
        <Box sx={{ mt: 1, textAlign: 'right' }}>
          <Typography variant="caption" color="text.secondary">
            {transcript.trim().split(/\s+/).length} words
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default TranscriptDisplay;