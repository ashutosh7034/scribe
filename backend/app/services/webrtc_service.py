"""
WebRTC Service

Real-time audio streaming service using WebRTC for low-latency
audio capture and transmission.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, AsyncGenerator
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class WebRTCConnection:
    """Manages individual WebRTC connection for audio streaming."""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.is_active = False
        self.audio_queue = asyncio.Queue()
        self.stats = {
            'bytes_received': 0,
            'packets_received': 0,
            'connection_start': datetime.now(),
            'last_packet_time': None
        }
    
    async def handle_audio_data(self, audio_data: bytes):
        """Handle incoming audio data from WebRTC stream."""
        if not self.is_active:
            return
            
        # Update statistics
        self.stats['bytes_received'] += len(audio_data)
        self.stats['packets_received'] += 1
        self.stats['last_packet_time'] = datetime.now()
        
        # Queue audio data for processing
        await self.audio_queue.put(audio_data)
    
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio stream as async generator."""
        while self.is_active:
            try:
                # Wait for audio data with timeout
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=1.0
                )
                yield audio_data
            except asyncio.TimeoutError:
                # Continue waiting for more data
                continue
            except Exception as e:
                logger.error(f"Error in audio stream: {e}")
                break
    
    def start(self):
        """Start the WebRTC connection."""
        self.is_active = True
        logger.info(f"WebRTC connection {self.connection_id} started")
    
    def stop(self):
        """Stop the WebRTC connection."""
        self.is_active = False
        logger.info(f"WebRTC connection {self.connection_id} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = self.stats.copy()
        if stats['last_packet_time']:
            stats['connection_duration'] = (
                datetime.now() - stats['connection_start']
            ).total_seconds()
            stats['last_packet_age'] = (
                datetime.now() - stats['last_packet_time']
            ).total_seconds()
        return stats


class WebRTCSignalingServer:
    """WebRTC signaling server for establishing peer connections."""
    
    def __init__(self):
        self.connections: Dict[str, WebRTCConnection] = {}
        self.signaling_handlers = {
            'offer': self._handle_offer,
            'answer': self._handle_answer,
            'ice-candidate': self._handle_ice_candidate,
            'audio-data': self._handle_audio_data
        }
    
    async def handle_signaling_message(self, 
                                     connection_id: str, 
                                     message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle WebRTC signaling messages.
        
        Args:
            connection_id: Unique connection identifier
            message: Signaling message from client
            
        Returns:
            Response message for client
        """
        message_type = message.get('type')
        
        if message_type not in self.signaling_handlers:
            return {'error': f'Unknown message type: {message_type}'}
        
        try:
            handler = self.signaling_handlers[message_type]
            return await handler(connection_id, message)
        except Exception as e:
            logger.error(f"Error handling signaling message: {e}")
            return {'error': str(e)}
    
    async def _handle_offer(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebRTC offer from client."""
        # Create new connection if it doesn't exist
        if connection_id not in self.connections:
            self.connections[connection_id] = WebRTCConnection(connection_id)
        
        connection = self.connections[connection_id]
        connection.start()
        
        # In a real implementation, you'd process the SDP offer
        # and generate an appropriate SDP answer
        # For now, we'll return a mock answer
        
        answer = {
            'type': 'answer',
            'sdp': 'mock-sdp-answer',  # In production, generate real SDP
            'connection_id': connection_id
        }
        
        logger.info(f"Generated answer for connection {connection_id}")
        return answer
    
    async def _handle_answer(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebRTC answer from client."""
        # Process SDP answer
        logger.info(f"Received answer for connection {connection_id}")
        return {'status': 'answer_processed'}
    
    async def _handle_ice_candidate(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ICE candidate from client."""
        candidate = message.get('candidate')
        logger.info(f"Received ICE candidate for connection {connection_id}: {candidate}")
        return {'status': 'candidate_processed'}
    
    async def _handle_audio_data(self, connection_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle audio data from WebRTC stream."""
        if connection_id not in self.connections:
            return {'error': 'Connection not found'}
        
        connection = self.connections[connection_id]
        audio_data = message.get('audio_data', b'')
        
        # Convert base64 audio data if needed
        if isinstance(audio_data, str):
            import base64
            audio_data = base64.b64decode(audio_data)
        
        await connection.handle_audio_data(audio_data)
        return {'status': 'audio_received'}
    
    def get_connection(self, connection_id: str) -> Optional[WebRTCConnection]:
        """Get WebRTC connection by ID."""
        return self.connections.get(connection_id)
    
    def close_connection(self, connection_id: str):
        """Close WebRTC connection."""
        if connection_id in self.connections:
            self.connections[connection_id].stop()
            del self.connections[connection_id]
            logger.info(f"Closed connection {connection_id}")
    
    def get_active_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active connections."""
        return {
            conn_id: conn.get_stats() 
            for conn_id, conn in self.connections.items() 
            if conn.is_active
        }


class AudioStreamManager:
    """Manages audio streams from WebRTC connections."""
    
    def __init__(self, signaling_server: WebRTCSignalingServer):
        self.signaling_server = signaling_server
        self.active_streams: Dict[str, asyncio.Task] = {}
    
    async def start_audio_processing(self, 
                                   connection_id: str,
                                   audio_processor: Callable[[AsyncGenerator[bytes, None]], None]):
        """
        Start processing audio stream from WebRTC connection.
        
        Args:
            connection_id: WebRTC connection ID
            audio_processor: Function to process audio stream
        """
        connection = self.signaling_server.get_connection(connection_id)
        if not connection:
            logger.error(f"Connection {connection_id} not found")
            return
        
        # Start audio processing task
        task = asyncio.create_task(
            audio_processor(connection.get_audio_stream())
        )
        self.active_streams[connection_id] = task
        
        logger.info(f"Started audio processing for connection {connection_id}")
    
    def stop_audio_processing(self, connection_id: str):
        """Stop audio processing for connection."""
        if connection_id in self.active_streams:
            task = self.active_streams[connection_id]
            task.cancel()
            del self.active_streams[connection_id]
            logger.info(f"Stopped audio processing for connection {connection_id}")
    
    def get_active_streams(self) -> Dict[str, bool]:
        """Get status of active audio streams."""
        return {
            conn_id: not task.done() 
            for conn_id, task in self.active_streams.items()
        }


class WebRTCService:
    """Main WebRTC service orchestrator."""
    
    def __init__(self):
        self.signaling_server = WebRTCSignalingServer()
        self.stream_manager = AudioStreamManager(self.signaling_server)
        self.service_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_audio_bytes': 0,
            'service_start_time': datetime.now()
        }
    
    async def handle_websocket_message(self, 
                                     connection_id: str, 
                                     message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle WebSocket message for WebRTC signaling.
        
        Args:
            connection_id: Connection identifier
            message: Message from WebSocket client
            
        Returns:
            Response message
        """
        # Update service stats
        if connection_id not in self.signaling_server.connections:
            self.service_stats['total_connections'] += 1
        
        # Handle signaling message
        response = await self.signaling_server.handle_signaling_message(
            connection_id, message
        )
        
        # Update active connections count
        self.service_stats['active_connections'] = len(
            self.signaling_server.get_active_connections()
        )
        
        return response
    
    async def start_audio_stream_processing(self, 
                                          connection_id: str,
                                          processor_callback: Callable[[AsyncGenerator[bytes, None]], None]):
        """Start processing audio stream from WebRTC connection."""
        await self.stream_manager.start_audio_processing(
            connection_id, processor_callback
        )
    
    def close_connection(self, connection_id: str):
        """Close WebRTC connection and cleanup resources."""
        self.stream_manager.stop_audio_processing(connection_id)
        self.signaling_server.close_connection(connection_id)
        
        # Update stats
        self.service_stats['active_connections'] = len(
            self.signaling_server.get_active_connections()
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        stats = self.service_stats.copy()
        stats.update({
            'connection_stats': self.signaling_server.get_active_connections(),
            'stream_stats': self.stream_manager.get_active_streams(),
            'uptime_seconds': (
                datetime.now() - self.service_stats['service_start_time']
            ).total_seconds()
        })
        return stats


# Global WebRTC service instance
webrtc_service = WebRTCService()