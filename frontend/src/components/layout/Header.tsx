/**
 * Application Header Component
 * 
 * Main navigation and branding for the Scribe application.
 */

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Person as PersonIcon,
  Accessibility as AccessibilityIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Header: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        {/* Logo and Title */}
        <Box style={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <AccessibilityIcon style={{ marginRight: 8 }} />
          <Typography
            variant="h6"
            component="h1"
            style={{ 
              fontWeight: 'bold',
              cursor: 'pointer',
            }}
            onClick={() => handleNavigation('/')}
          >
            Scribe
          </Typography>
          <Typography
            variant="body2"
            style={{ 
              marginLeft: 8, 
              opacity: 0.8,
            }}
          >
            Real-time Sign Language Translation
          </Typography>
        </Box>

        {/* Navigation Buttons */}
        <Box style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Button
            color="inherit"
            onClick={() => handleNavigation('/')}
            style={{ 
              textTransform: 'none',
              backgroundColor: location.pathname === '/' ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Translate
          </Button>
          
          <Button
            color="inherit"
            onClick={() => handleNavigation('/avatars')}
            style={{ 
              textTransform: 'none',
              backgroundColor: location.pathname === '/avatars' ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            Avatars
          </Button>

          <IconButton
            color="inherit"
            onClick={() => handleNavigation('/settings')}
            aria-label="Settings"
            style={{
              backgroundColor: location.pathname === '/settings' ? 'rgba(255,255,255,0.1)' : 'transparent'
            }}
          >
            <SettingsIcon />
          </IconButton>

          {/* User Profile (placeholder for future authentication) */}
          <IconButton
            color="inherit"
            aria-label="User profile"
            style={{ marginLeft: 8 }}
          >
            <PersonIcon />
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;