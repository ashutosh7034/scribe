/**
 * Avatar Customization Page
 * 
 * Page for selecting and customizing 3D avatars.
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
} from '@mui/material';

const AvatarCustomizationPage: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Avatar Customization
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Choose and customize your sign language avatar. Personalize appearance, 
        signing style, and preferences to match your needs.
      </Typography>

      <Grid container spacing={3}>
        {/* Avatar Gallery */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Available Avatars
            </Typography>
            
            <Grid container spacing={2}>
              {/* Placeholder avatar cards */}
              {[1, 2, 3, 4].map((id) => (
                <Grid item xs={12} sm={6} md={4} key={id}>
                  <Card>
                    <CardMedia
                      sx={{ height: 200, backgroundColor: 'grey.200' }}
                      title={`Avatar ${id}`}
                    />
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Avatar {id}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Professional signing style
                      </Typography>
                      <Button 
                        variant="outlined" 
                        size="small" 
                        sx={{ mt: 1 }}
                      >
                        Select
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Customization Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Customization Options
            </Typography>
            
            <Typography variant="body2" color="text.secondary">
              Select an avatar to customize appearance and signing preferences.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AvatarCustomizationPage;