/**
 * Main Application Component
 * 
 * Root component that handles routing and global application state.
 */

import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Container } from '@mui/material';

import Header from '@/components/layout/Header';
import TranslationPage from '@/pages/TranslationPage';
import AvatarCustomizationPage from '@/pages/AvatarCustomizationPage';
import SettingsPage from '@/pages/SettingsPage';
import { useAppStore } from '@/store/appStore';

function App() {
  const { highContrastMode, fontScaling } = useAppStore();

  return (
    <div 
      className={`app ${highContrastMode ? 'high-contrast' : ''} font-scale-${fontScaling}`}
      role="application"
      aria-label="Scribe Sign Language Translation Platform"
    >
      {/* Skip link for keyboard navigation */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <Header />

      <main id="main-content" role="main">
        <Container maxWidth="xl" sx={{ py: 2 }}>
          <Routes>
            <Route path="/" element={<TranslationPage />} />
            <Route path="/avatars" element={<AvatarCustomizationPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Container>
      </main>
    </div>
  );
}

export default App;