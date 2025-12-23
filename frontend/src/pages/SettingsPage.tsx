/**
 * Settings Page
 * 
 * User preferences and application settings.
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Switch,
  Slider,
  Divider,
} from '@mui/material';
import { useAppStore } from '@/store/appStore';

const SettingsPage: React.FC = () => {
  const {
    highContrastMode,
    fontScaling,
    setHighContrastMode,
    setFontScaling,
  } = useAppStore();

  const handleFontScalingChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFontScaling(event.target.value as any);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Customize your Scribe experience with accessibility options and preferences.
      </Typography>

      <Grid container spacing={3}>
        {/* Accessibility Settings */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Accessibility
            </Typography>

            {/* High Contrast Mode */}
            <Box sx={{ mb: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={highContrastMode}
                    onChange={(e) => setHighContrastMode(e.target.checked)}
                  />
                }
                label="High Contrast Mode"
              />
              <Typography variant="body2" color="text.secondary">
                Increase contrast for better visibility
              </Typography>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Font Scaling */}
            <Box sx={{ mb: 3 }}>
              <FormControl component="fieldset">
                <FormLabel component="legend">Font Size</FormLabel>
                <RadioGroup
                  value={fontScaling}
                  onChange={handleFontScalingChange}
                >
                  <FormControlLabel value="small" control={<Radio />} label="Small" />
                  <FormControlLabel value="normal" control={<Radio />} label="Normal" />
                  <FormControlLabel value="large" control={<Radio />} label="Large" />
                  <FormControlLabel value="extra-large" control={<Radio />} label="Extra Large" />
                </RadioGroup>
              </FormControl>
            </Box>
          </Paper>
        </Grid>

        {/* Translation Settings */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Translation Preferences
            </Typography>

            {/* Sign Language Selection */}
            <Box sx={{ mb: 3 }}>
              <FormControl component="fieldset">
                <FormLabel component="legend">Sign Language</FormLabel>
                <RadioGroup defaultValue="ASL">
                  <FormControlLabel value="ASL" control={<Radio />} label="American Sign Language (ASL)" />
                  <FormControlLabel value="BSL" control={<Radio />} label="British Sign Language (BSL)" />
                  <FormControlLabel value="Auslan" control={<Radio />} label="Australian Sign Language (Auslan)" />
                </RadioGroup>
              </FormControl>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Signing Speed */}
            <Box sx={{ mb: 3 }}>
              <Typography gutterBottom>
                Signing Speed
              </Typography>
              <Slider
                defaultValue={100}
                min={50}
                max={200}
                step={10}
                marks={[
                  { value: 50, label: 'Slow' },
                  { value: 100, label: 'Normal' },
                  { value: 200, label: 'Fast' },
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}%`}
              />
            </Box>

            {/* Emotion Intensity */}
            <Box sx={{ mb: 3 }}>
              <Typography gutterBottom>
                Emotion Intensity
              </Typography>
              <Slider
                defaultValue={100}
                min={0}
                max={200}
                step={10}
                marks={[
                  { value: 0, label: 'None' },
                  { value: 100, label: 'Normal' },
                  { value: 200, label: 'High' },
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}%`}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SettingsPage;