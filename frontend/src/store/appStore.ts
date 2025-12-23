/**
 * Global Application Store
 * 
 * Zustand-based state management for the Scribe application.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { AppState, Avatar, TranslationSession, ScribeError } from '../types';

interface AppStore extends AppState {
  // Actions
  setHighContrastMode: (enabled: boolean) => void;
  setFontScaling: (scale: 'small' | 'normal' | 'large' | 'extra-large') => void;
  setSelectedAvatar: (avatar: Avatar) => void;
  setTranslating: (isTranslating: boolean) => void;
  setMicrophoneActive: (isActive: boolean) => void;
  setAudioLevel: (level: number) => void;
  setConnectionState: (isConnected: boolean, quality?: 'excellent' | 'good' | 'fair' | 'poor') => void;
  setCurrentSession: (session: TranslationSession | undefined) => void;
  setError: (error: string | ScribeError | undefined) => void;
  setLoadingStates: (states: Partial<Pick<AppState, 'isLoadingAvatars' | 'isLoadingTranslation'>>) => void;
  
  // Reset functions
  resetTranslationState: () => void;
  resetError: () => void;
}

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // Initial state
      highContrastMode: false,
      fontScaling: 'normal',
      isTranslating: false,
      currentSession: undefined,
      selectedAvatar: undefined,
      isMicrophoneActive: false,
      audioLevel: 0,
      isConnected: false,
      connectionQuality: 'good',
      error: undefined,
      isLoadingAvatars: false,
      isLoadingTranslation: false,

      // Actions
      setHighContrastMode: (enabled) => 
        set({ highContrastMode: enabled }),

      setFontScaling: (scale) => 
        set({ fontScaling: scale }),

      setSelectedAvatar: (avatar) => 
        set({ selectedAvatar: avatar }),

      setTranslating: (isTranslating) => 
        set({ isTranslating }),

      setMicrophoneActive: (isActive) => 
        set({ isMicrophoneActive: isActive }),

      setAudioLevel: (level) => 
        set({ audioLevel: Math.max(0, Math.min(100, level)) }),

      setConnectionState: (isConnected, quality = 'good') => 
        set({ isConnected, connectionQuality: quality }),

      setCurrentSession: (session) => 
        set({ currentSession: session }),

      setError: (error) => {
        if (typeof error === 'string') {
          set({ error });
        } else if (error instanceof Error) {
          set({ error: error.message });
        } else {
          set({ error: undefined });
        }
      },

      setLoadingStates: (states) => 
        set((state) => ({ ...state, ...states })),

      // Reset functions
      resetTranslationState: () => 
        set({
          isTranslating: false,
          currentSession: undefined,
          isMicrophoneActive: false,
          audioLevel: 0,
          error: undefined,
        }),

      resetError: () => 
        set({ error: undefined }),
    }),
    {
      name: 'scribe-app-store',
      // Only persist user preferences, not session state
      partialize: (state) => ({
        highContrastMode: state.highContrastMode,
        fontScaling: state.fontScaling,
        selectedAvatar: state.selectedAvatar,
      }),
    }
  )
);