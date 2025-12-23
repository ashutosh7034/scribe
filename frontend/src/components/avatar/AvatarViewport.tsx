/**
 * Avatar Viewport Component
 * 
 * 3D avatar rendering viewport using Three.js and React Three Fiber.
 */

import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei';
import { AvatarViewportProps } from '../../types';
import Avatar3D from './Avatar3D';

const AvatarViewport: React.FC<AvatarViewportProps> = ({
  avatar,
  poseSequence = [],
  facialExpressions = [],
  isPlaying,
  quality = 'standard',
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Quality settings
  const qualitySettings = {
    low: { pixelRatio: 0.5, shadows: false, antialias: false },
    standard: { pixelRatio: 1, shadows: true, antialias: true },
    high: { pixelRatio: Math.min(window.devicePixelRatio, 2), shadows: true, antialias: true }
  };

  const settings = qualitySettings[quality];

  useEffect(() => {
    if (avatar) {
      setIsLoading(true);
      setError(null);
      // Simulate loading time for avatar assets
      const timer = setTimeout(() => {
        setIsLoading(false);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [avatar]);

  const handleAvatarError = (error: Error) => {
    console.error('Avatar rendering error:', error);
    setError('Failed to load avatar');
    setIsLoading(false);
  };

  const handleAvatarLoaded = () => {
    setIsLoading(false);
    setError(null);
  };

  if (!avatar) {
    return (
      <Box
        className="avatar-viewport"
        sx={{
          width: '100%',
          height: '100%',
          minHeight: '300px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'grey.100',
          borderRadius: 1,
        }}
      >
        <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
          <Typography variant="h6" gutterBottom>
            No Avatar Selected
          </Typography>
          <Typography variant="body2">
            Please select an avatar to begin
          </Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box
      className="avatar-viewport"
      sx={{
        width: '100%',
        height: '100%',
        minHeight: '300px',
        position: 'relative',
        backgroundColor: 'grey.50',
        borderRadius: 1,
        overflow: 'hidden',
      }}
    >
      {/* Loading overlay */}
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            zIndex: 10,
          }}
        >
          <Box sx={{ textAlign: 'center' }}>
            <CircularProgress size={40} />
            <Typography variant="body2" sx={{ mt: 2 }}>
              Loading Avatar...
            </Typography>
          </Box>
        </Box>
      )}

      {/* Error overlay */}
      {error && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            zIndex: 10,
          }}
        >
          <Box sx={{ textAlign: 'center', color: 'error.main' }}>
            <Typography variant="h6" gutterBottom>
              Avatar Error
            </Typography>
            <Typography variant="body2">
              {error}
            </Typography>
          </Box>
        </Box>
      )}

      {/* 3D Canvas */}
      <Canvas
        dpr={settings.pixelRatio}
        shadows={settings.shadows}
        gl={{ 
          antialias: settings.antialias,
          alpha: true,
          powerPreference: 'high-performance'
        }}
        style={{ width: '100%', height: '100%' }}
      >
        <PerspectiveCamera makeDefault position={[0, 1.6, 3]} fov={50} />
        
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[5, 5, 5]}
          intensity={0.8}
          castShadow={settings.shadows}
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <pointLight position={[-2, 2, 2]} intensity={0.3} />

        {/* Environment */}
        <Environment preset="studio" />

        {/* Avatar */}
        <Suspense fallback={null}>
          <Avatar3D
            avatar={avatar}
            poseSequence={poseSequence}
            facialExpressions={facialExpressions}
            isPlaying={isPlaying}
            onError={handleAvatarError}
            onLoaded={handleAvatarLoaded}
          />
        </Suspense>

        {/* Ground plane */}
        <mesh 
          rotation={[-Math.PI / 2, 0, 0]} 
          position={[0, -0.1, 0]}
          receiveShadow={settings.shadows}
        >
          <planeGeometry args={[10, 10]} />
          <meshStandardMaterial color="#f5f5f5" transparent opacity={0.5} />
        </mesh>

        {/* Controls */}
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={8}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2}
          target={[0, 1.2, 0]}
        />
      </Canvas>

      {/* Status overlay */}
      <Box
        sx={{
          position: 'absolute',
          top: 8,
          left: 8,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          px: 1,
          py: 0.5,
          borderRadius: 1,
          fontSize: '0.75rem',
        }}
      >
        {isPlaying ? 'Playing' : 'Idle'} | {quality.toUpperCase()}
        {poseSequence.length > 0 && ` | ${poseSequence.length} poses`}
      </Box>
    </Box>
  );
};

export default AvatarViewport;