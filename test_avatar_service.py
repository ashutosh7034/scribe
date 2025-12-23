#!/usr/bin/env python3
"""
Simple test script for avatar rendering service
"""

import sys
import os

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from app.services.avatar_rendering import avatar_rendering_service
    print("✓ Avatar service imported successfully")
    
    # Test basic functionality
    pose_sequence = avatar_rendering_service.generate_pose_sequence(
        text="hello world",
        emotion="excitement",
        emotion_intensity=0.7,
        signing_speed=1.0
    )
    
    print(f"✓ Generated {len(pose_sequence)} poses")
    
    if pose_sequence:
        first_pose = pose_sequence[0]
        print(f"✓ First pose timestamp: {first_pose.timestamp}ms")
        print(f"✓ First pose has {len(first_pose.joints)} joints")
        
        # Test validation
        is_valid = avatar_rendering_service.validate_pose_sequence(pose_sequence)
        print(f"✓ Pose sequence validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test performance metrics
        metrics = avatar_rendering_service.get_avatar_performance_metrics(pose_sequence)
        print(f"✓ Performance metrics: {metrics}")
    
    print("\n🎉 Avatar rendering service test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing avatar service: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)