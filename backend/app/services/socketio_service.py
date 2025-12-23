"""
Socket.IO Service

Handles real-time communication with frontend using Socket.IO.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import socketio
from fastapi import FastAPI

logger = logging.getLogger(__name__)

class SocketIOService:
    """Socket.IO service for real-time communication."""
    
    def __init__(self):
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True
        )
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def get_app(self) -> socketio.ASGIApp:
        """Get the Socket.IO ASGI app."""
        return socketio.ASGIApp(self.sio)
    
    def setup_events(self):
        """Set up Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection."""
            logger.info(f"Socket.IO client connected: {sid}")
            await self.sio.emit('connected', {'status': 'success'}, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            logger.info(f"Socket.IO client disconnected: {sid}")
            # Clean up any active sessions for this client
            sessions_to_remove = []
            for session_id, session_data in self.active_sessions.items():
                if session_data.get('socket_id') == sid:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up session {session_id} for disconnected client {sid}")
        
        @self.sio.event
        async def start_translation_session(sid, data):
            """Start a new translation session."""
            try:
                session_id = data.get('session_id')
                if not session_id:
                    await self.sio.emit('session_error', {
                        'error': 'Missing session_id'
                    }, room=sid)
                    return
                
                # Store session info
                self.active_sessions[session_id] = {
                    'socket_id': sid,
                    'config': data.get('config', {}),
                    'started_at': data.get('timestamp'),
                    'status': 'active'
                }
                
                logger.info(f"Started translation session {session_id} for client {sid}")
                
                await self.sio.emit('session_started', {
                    'session_id': session_id,
                    'status': 'success'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error starting session: {e}")
                await self.sio.emit('session_error', {
                    'error': str(e)
                }, room=sid)
        
        @self.sio.event
        async def end_translation_session(sid, data):
            """End a translation session."""
            try:
                session_id = data.get('session_id')
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                    logger.info(f"Ended translation session {session_id}")
                
                await self.sio.emit('session_ended', {
                    'session_id': session_id,
                    'status': 'success'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error ending session: {e}")
                await self.sio.emit('session_error', {
                    'error': str(e)
                }, room=sid)
        
        @self.sio.event
        async def audio_data(sid, data):
            """Handle incoming audio data."""
            try:
                session_id = data.get('session_id')
                if session_id not in self.active_sessions:
                    await self.sio.emit('session_error', {
                        'session_id': session_id,
                        'error': 'Session not found'
                    }, room=sid)
                    return
                
                # For now, just acknowledge receipt
                # In a full implementation, this would process the audio
                logger.debug(f"Received audio data for session {session_id}")
                
                # Send mock translation frame back
                await self.sio.emit('translation_frame', {
                    'session_id': session_id,
                    'timestamp': data.get('timestamp'),
                    'pose_data': [],  # Mock pose data
                    'facial_expression': 'neutral'
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                await self.sio.emit('session_error', {
                    'error': str(e)
                }, room=sid)
        
        @self.sio.event
        async def translate_text(sid, data):
            """Handle text translation request."""
            try:
                session_id = data.get('session_id')
                text = data.get('text', '')
                
                if session_id not in self.active_sessions:
                    await self.sio.emit('session_error', {
                        'session_id': session_id,
                        'error': 'Session not found'
                    }, room=sid)
                    return
                
                logger.info(f"Translating text for session {session_id}: {text}")
                
                # Send mock transcript update
                await self.sio.emit('transcript_update', {
                    'session_id': session_id,
                    'transcript': text
                }, room=sid)
                
                # Send mock translation complete
                await self.sio.emit('translation_complete', {
                    'sessionId': session_id,
                    'transcript': text,
                    'poseSequence': [],  # Mock pose sequence
                    'facialExpressions': [],  # Mock facial expressions
                    'confidence': 0.95,
                    'processingTime': 100
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error translating text: {e}")
                await self.sio.emit('session_error', {
                    'error': str(e)
                }, room=sid)
    
    async def emit_to_session(self, session_id: str, event: str, data: Dict[str, Any]):
        """Emit an event to a specific session."""
        if session_id in self.active_sessions:
            socket_id = self.active_sessions[session_id]['socket_id']
            await self.sio.emit(event, data, room=socket_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions."""
        return self.active_sessions.copy()

# Global instance
socketio_service = SocketIOService()
socketio_service.setup_events()