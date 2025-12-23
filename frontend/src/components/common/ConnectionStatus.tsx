/**
 * Connection Status Component
 * 
 * Displays current connection status and quality indicator.
 */

import React from 'react';
import {
  Chip,
  Typography,
} from '@mui/material';
import {
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon,
  SignalWifi1Bar,
  SignalWifi2Bar,
  SignalWifi3Bar,
  SignalWifi4Bar,
} from '@mui/icons-material';

interface ConnectionStatusProps {
  isConnected: boolean;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
  style?: React.CSSProperties;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  quality,
  style,
}) => {
  const getQualityIcon = () => {
    if (!isConnected) return <WifiOffIcon />;
    
    switch (quality) {
      case 'excellent':
        return <SignalWifi4Bar />;
      case 'good':
        return <SignalWifi3Bar />;
      case 'fair':
        return <SignalWifi2Bar />;
      case 'poor':
        return <SignalWifi1Bar />;
      default:
        return <WifiIcon />;
    }
  };

  const getQualityColor = (): 'success' | 'warning' | 'error' | 'default' => {
    if (!isConnected) return 'error';
    
    switch (quality) {
      case 'excellent':
      case 'good':
        return 'success';
      case 'fair':
        return 'warning';
      case 'poor':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusText = () => {
    if (!isConnected) return 'Disconnected';
    return `Connected (${quality})`;
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        ...style,
      }}
    >
      <Chip
        icon={getQualityIcon()}
        label={getStatusText()}
        color={getQualityColor()}
        variant={isConnected ? 'filled' : 'outlined'}
        size="small"
      />
      
      {!isConnected && (
        <Typography variant="caption" color="error.main">
          Check your internet connection
        </Typography>
      )}
    </div>
  );
};

export default ConnectionStatus;