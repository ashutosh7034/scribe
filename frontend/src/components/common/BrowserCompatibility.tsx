/**
 * Browser Compatibility Component
 * 
 * Checks browser compatibility and displays warnings for unsupported features.
 */

import React, { useEffect, useState } from 'react';
import { Alert, AlertTitle, Typography } from '@mui/material';

interface BrowserCompatibilityProps {
  onCompatibilityCheck?: (isCompatible: boolean) => void;
}

const BrowserCompatibility: React.FC<BrowserCompatibilityProps> = ({
  onCompatibilityCheck,
}) => {
  const [compatibilityIssues, setCompatibilityIssues] = useState<string[]>([]);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkCompatibility = () => {
      const issues: string[] = [];

      // Check WebRTC support
      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        issues.push('WebRTC audio capture is not supported');
      }

      // Check MediaRecorder support
      if (!window.MediaRecorder) {
        issues.push('Audio recording is not supported');
      }

      // Check WebGL support
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) {
        issues.push('WebGL is not supported (required for 3D avatar)');
      }

      // Check WebSocket support
      if (!window.WebSocket) {
        issues.push('WebSocket is not supported');
      }

      // Check AudioContext support
      if (!window.AudioContext && !(window as any).webkitAudioContext) {
        issues.push('Web Audio API is not supported');
      }

      setCompatibilityIssues(issues);
      setIsChecking(false);
      onCompatibilityCheck?.(issues.length === 0);
    };

    checkCompatibility();
  }, [onCompatibilityCheck]);

  if (isChecking) {
    return null;
  }

  if (compatibilityIssues.length === 0) {
    return null;
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <Alert severity="warning">
        <AlertTitle>Browser Compatibility Issues</AlertTitle>
        <Typography variant="body2" gutterBottom>
          Your browser may not support all features of Scribe:
        </Typography>
        <ul style={{ margin: 0, paddingLeft: '20px' }}>
          {compatibilityIssues.map((issue, index) => (
            <li key={index}>
              <Typography variant="body2">{issue}</Typography>
            </li>
          ))}
        </ul>
        <Typography variant="body2" style={{ marginTop: 8 }}>
          For the best experience, please use a modern browser like Chrome, Firefox, or Safari.
        </Typography>
      </Alert>
    </div>
  );
};

export default BrowserCompatibility;