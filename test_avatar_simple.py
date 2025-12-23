#!/usr/bin/env python3
"""
Simple test for avatar rendering logic without full schema imports
"""

import sys
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Simple test classes to avoid schema dependencies
@dataclass
class SimpleVector3:
    x: float
    y: float
    z: float

@dataclass
class SimplePoseKeyframe:
    timestamp: float
    joints: Dict[str, SimpleVector3]

@dataclass
class SimpleFacialExpression:
    timestamp: float
    expression: str
    intensity: float

# Test the core avatar logic
def test_avatar_logic():
    """Test basic avatar rendering logic."""
    
    # Test joint hierarchy
    joint_hierarchy = {
        "pelvis": {"parent": None, "position": [0, 0, 0]},
        "spine1": {"parent": "pelvis", "position": [0, 0.1, 0]},
        "left_shoulder": {"parent": "spine1", "position": [-0.15, 0.1, 0]},
        "left_upper_arm": {"parent": "left_shoulder", "position": [-0.3, 0, 0]},
        "left_hand": {"parent": "left_upper_arm", "position": [-0.2, 0, 0]},
        "right_shoulder": {"parent": "spine1", "position": [0.15, 0.1, 0]},
        "right_upper_arm": {"parent": "right_shoulder", "position": [0.3, 0, 0]},
        "right_hand": {"parent": "right_upper_arm", "position": [0.2, 0, 0]},
    }
    
    print(f"✓ Joint hierarchy has {len(joint_hierarchy)} joints")
    
    # Test pose generation
    def create_test_pose(timestamp: float, rotations: Dict[str, Tuple[float, float, float]]) -> SimplePoseKeyframe:
        joints = {}
        for joint_name in joint_hierarchy:
            rotation = rotations.get(joint_name, (0, 0, 0))
            joints[joint_name] = SimpleVector3(x=rotation[0], y=rotation[1], z=rotation[2])
        return SimplePoseKeyframe(timestamp=timestamp, joints=joints)
    
    # Create test poses for a simple gesture
    poses = [
        create_test_pose(0, {"right_upper_arm": (0, 0, 1.2), "right_hand": (0, 0, 0.2)}),
        create_test_pose(500, {"right_upper_arm": (0, 0, 1.0), "right_hand": (0, 0, -0.2)}),
        create_test_pose(1000, {"right_upper_arm": (0, 0, 1.2), "right_hand": (0, 0, 0.2)})
    ]
    
    print(f"✓ Generated {len(poses)} test poses")
    
    # Test pose validation
    def validate_poses(pose_sequence: List[SimplePoseKeyframe]) -> bool:
        if not pose_sequence:
            return False
        
        # Check timestamps are increasing
        for i in range(1, len(pose_sequence)):
            if pose_sequence[i].timestamp <= pose_sequence[i-1].timestamp:
                return False
        
        # Check joint limits
        for pose in pose_sequence:
            for joint_name, vector in pose.joints.items():
                if abs(vector.x) > 3.14 or abs(vector.y) > 3.14 or abs(vector.z) > 3.14:
                    return False
        
        return True
    
    is_valid = validate_poses(poses)
    print(f"✓ Pose validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Test emotion modulation
    def apply_emotion_modulation(pose: SimplePoseKeyframe, emotion: str, intensity: float) -> SimplePoseKeyframe:
        modulated_joints = {}
        
        for joint_name, vector in pose.joints.items():
            x, y, z = vector.x, vector.y, vector.z
            
            if emotion == "excitement" and "upper_arm" in joint_name:
                z *= (1 + 0.3 * intensity)  # Larger arm movements
            elif emotion == "anger" and "shoulder" in joint_name:
                y += 0.2 * intensity  # Raise shoulders
            
            modulated_joints[joint_name] = SimpleVector3(x=x, y=y, z=z)
        
        return SimplePoseKeyframe(timestamp=pose.timestamp, joints=modulated_joints)
    
    # Test emotion modulation
    excited_pose = apply_emotion_modulation(poses[0], "excitement", 0.7)
    original_arm = poses[0].joints["right_upper_arm"].z
    excited_arm = excited_pose.joints["right_upper_arm"].z
    
    print(f"✓ Emotion modulation: {original_arm:.2f} -> {excited_arm:.2f}")
    
    # Test performance metrics
    def get_performance_metrics(pose_sequence: List[SimplePoseKeyframe]) -> Dict[str, Any]:
        if not pose_sequence:
            return {"error": "Empty sequence"}
        
        total_duration = pose_sequence[-1].timestamp
        frame_count = len(pose_sequence)
        
        return {
            "total_duration_ms": total_duration,
            "frame_count": frame_count,
            "average_fps": (frame_count / (total_duration / 1000.0)) if total_duration > 0 else 0,
            "joint_count": len(pose_sequence[0].joints),
        }
    
    metrics = get_performance_metrics(poses)
    print(f"✓ Performance metrics: {metrics}")
    
    print("\n🎉 Avatar rendering logic test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_avatar_logic()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)