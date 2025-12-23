/**
 * Avatar3D Component
 * 
 * 3D avatar rendering with skeletal animation and sign language gesture playback.
 * Implements basic SMPL-X skeleton with hand and arm movement animations.
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { 
  Group, 
  SkinnedMesh, 
  Skeleton, 
  Bone, 
  Vector3, 
  Quaternion,
  AnimationMixer,
  AnimationClip,
  KeyframeTrack,
  InterpolateLinear,
  Euler
} from 'three';
import { Avatar, PoseKeyframe, FacialExpressionKeyframe } from '../../types';

interface Avatar3DProps {
  avatar: Avatar;
  poseSequence?: PoseKeyframe[];
  facialExpressions?: FacialExpressionKeyframe[];
  isPlaying: boolean;
  onError?: (error: Error) => void;
  onLoaded?: () => void;
}

// Basic SMPL-X joint hierarchy for sign language
const JOINT_HIERARCHY = {
  // Torso
  pelvis: { parent: null, position: [0, 0, 0] },
  spine1: { parent: 'pelvis', position: [0, 0.1, 0] },
  spine2: { parent: 'spine1', position: [0, 0.15, 0] },
  spine3: { parent: 'spine2', position: [0, 0.15, 0] },
  neck: { parent: 'spine3', position: [0, 0.2, 0] },
  head: { parent: 'neck', position: [0, 0.15, 0] },
  
  // Left arm
  left_shoulder: { parent: 'spine3', position: [-0.15, 0.1, 0] },
  left_upper_arm: { parent: 'left_shoulder', position: [-0.3, 0, 0] },
  left_forearm: { parent: 'left_upper_arm', position: [-0.25, 0, 0] },
  left_hand: { parent: 'left_forearm', position: [-0.2, 0, 0] },
  
  // Right arm
  right_shoulder: { parent: 'spine3', position: [0.15, 0.1, 0] },
  right_upper_arm: { parent: 'right_shoulder', position: [0.3, 0, 0] },
  right_forearm: { parent: 'right_upper_arm', position: [0.25, 0, 0] },
  right_hand: { parent: 'right_forearm', position: [0.2, 0, 0] },
  
  // Hand joints (simplified - key joints for sign language)
  left_thumb: { parent: 'left_hand', position: [-0.05, 0.02, 0.02] },
  left_index: { parent: 'left_hand', position: [-0.08, 0.01, 0] },
  left_middle: { parent: 'left_hand', position: [-0.08, 0, 0] },
  left_ring: { parent: 'left_hand', position: [-0.08, -0.01, 0] },
  left_pinky: { parent: 'left_hand', position: [-0.08, -0.02, 0] },
  
  right_thumb: { parent: 'right_hand', position: [0.05, 0.02, 0.02] },
  right_index: { parent: 'right_hand', position: [0.08, 0.01, 0] },
  right_middle: { parent: 'right_hand', position: [0.08, 0, 0] },
  right_ring: { parent: 'right_hand', position: [0.08, -0.01, 0] },
  right_pinky: { parent: 'right_hand', position: [0.08, -0.02, 0] },
};

// Default pose for neutral standing position
const DEFAULT_POSE = {
  pelvis: { rotation: [0, 0, 0], position: [0, 1, 0] },
  spine1: { rotation: [0, 0, 0] },
  spine2: { rotation: [0, 0, 0] },
  spine3: { rotation: [0, 0, 0] },
  neck: { rotation: [0, 0, 0] },
  head: { rotation: [0, 0, 0] },
  left_shoulder: { rotation: [0, 0, -0.1] },
  left_upper_arm: { rotation: [0, 0, -0.2] },
  left_forearm: { rotation: [0, 0, -0.3] },
  left_hand: { rotation: [0, 0, 0] },
  right_shoulder: { rotation: [0, 0, 0.1] },
  right_upper_arm: { rotation: [0, 0, 0.2] },
  right_forearm: { rotation: [0, 0, 0.3] },
  right_hand: { rotation: [0, 0, 0] },
};

const Avatar3D: React.FC<Avatar3DProps> = ({
  avatar,
  poseSequence = [],
  facialExpressions = [],
  isPlaying,
  onError,
  onLoaded
}) => {
  const groupRef = useRef<Group>(null);
  const skeletonRef = useRef<Skeleton | null>(null);
  const mixerRef = useRef<AnimationMixer | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [currentPoseIndex, setCurrentPoseIndex] = useState(0);
  const [animationTime, setAnimationTime] = useState(0);

  // Create basic avatar geometry (simplified humanoid)
  const avatarGeometry = useMemo(() => {
    // Create bones for the skeleton
    const bones: Bone[] = [];
    const boneMap: Record<string, Bone> = {};

    // Create all bones
    Object.entries(JOINT_HIERARCHY).forEach(([name, config]) => {
      const bone = new Bone();
      bone.name = name;
      bone.position.set(config.position[0], config.position[1], config.position[2]);
      bones.push(bone);
      boneMap[name] = bone;
    });

    // Set up parent-child relationships
    Object.entries(JOINT_HIERARCHY).forEach(([name, config]) => {
      if (config.parent) {
        boneMap[config.parent].add(boneMap[name]);
      }
    });

    // Create skeleton
    const skeleton = new Skeleton(bones);
    skeletonRef.current = skeleton;

    return skeleton;
  }, []);

  // Apply pose to skeleton
  const applyPose = (pose: Record<string, any>) => {
    if (!skeletonRef.current) return;

    skeletonRef.current.bones.forEach(bone => {
      const poseData = pose[bone.name] || DEFAULT_POSE[bone.name as keyof typeof DEFAULT_POSE];
      
      if (poseData) {
        // Apply rotation
        if (poseData.rotation) {
          bone.rotation.set(
            poseData.rotation[0],
            poseData.rotation[1],
            poseData.rotation[2]
          );
        }
        
        // Apply position (mainly for root bone)
        if (poseData.position && bone.name === 'pelvis') {
          bone.position.set(
            poseData.position[0],
            poseData.position[1],
            poseData.position[2]
          );
        }
      }
    });
  };

  // Convert PoseKeyframe to internal pose format
  const convertPoseKeyframe = (keyframe: PoseKeyframe) => {
    const pose: Record<string, any> = {};
    
    Object.entries(keyframe.joints).forEach(([jointName, vector]) => {
      // Convert Vector3 positions to rotations for simplicity
      // In a real implementation, this would use proper IK solving
      pose[jointName] = {
        rotation: [
          vector.x * 0.1, // Scale down for reasonable rotations
          vector.y * 0.1,
          vector.z * 0.1
        ]
      };
    });
    
    return pose;
  };

  // Animation loop
  useFrame((state, delta) => {
    if (!isPlaying || !poseSequence.length) {
      // Apply default pose when not playing
      applyPose(DEFAULT_POSE);
      return;
    }

    // Update animation time
    setAnimationTime(prev => prev + delta * 1000); // Convert to milliseconds

    // Find current pose based on timestamp
    const currentTime = animationTime % (poseSequence[poseSequence.length - 1]?.timestamp || 1000);
    
    let currentIndex = 0;
    for (let i = 0; i < poseSequence.length; i++) {
      if (poseSequence[i].timestamp <= currentTime) {
        currentIndex = i;
      } else {
        break;
      }
    }

    // Apply interpolation between poses for smooth animation
    const currentKeyframe = poseSequence[currentIndex];
    const nextKeyframe = poseSequence[currentIndex + 1];

    if (currentKeyframe) {
      let pose = convertPoseKeyframe(currentKeyframe);

      // Simple linear interpolation if next keyframe exists
      if (nextKeyframe && currentTime < nextKeyframe.timestamp) {
        const t = (currentTime - currentKeyframe.timestamp) / 
                  (nextKeyframe.timestamp - currentKeyframe.timestamp);
        
        const nextPose = convertPoseKeyframe(nextKeyframe);
        
        // Interpolate rotations
        Object.keys(pose).forEach(jointName => {
          if (pose[jointName] && nextPose[jointName]) {
            pose[jointName].rotation = pose[jointName].rotation.map((val: number, idx: number) => 
              val + (nextPose[jointName].rotation[idx] - val) * t
            );
          }
        });
      }

      applyPose(pose);
    }

    setCurrentPoseIndex(currentIndex);
  });

  // Handle loading
  useEffect(() => {
    if (avatar && !isLoaded) {
      // Simulate loading time for avatar assets
      const timer = setTimeout(() => {
        setIsLoaded(true);
        onLoaded?.();
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [avatar, isLoaded, onLoaded]);

  // Reset animation when pose sequence changes
  useEffect(() => {
    setAnimationTime(0);
    setCurrentPoseIndex(0);
  }, [poseSequence]);

  if (!avatar || !isLoaded) {
    return null;
  }

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* Basic humanoid mesh with skeleton */}
      <skinnedMesh skeleton={avatarGeometry}>
        {/* Head */}
        <mesh position={[0, 1.7, 0]}>
          <sphereGeometry args={[0.12, 16, 16]} />
          <meshStandardMaterial color="#fdbcb4" />
        </mesh>
        
        {/* Torso */}
        <mesh position={[0, 1.2, 0]}>
          <boxGeometry args={[0.3, 0.5, 0.15]} />
          <meshStandardMaterial color="#4a90e2" />
        </mesh>
        
        {/* Left arm */}
        <group>
          <mesh position={[-0.2, 1.4, 0]}>
            <boxGeometry args={[0.08, 0.3, 0.08]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
          <mesh position={[-0.45, 1.1, 0]}>
            <boxGeometry args={[0.06, 0.25, 0.06]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
          <mesh position={[-0.6, 0.9, 0]}>
            <boxGeometry args={[0.08, 0.05, 0.15]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
        </group>
        
        {/* Right arm */}
        <group>
          <mesh position={[0.2, 1.4, 0]}>
            <boxGeometry args={[0.08, 0.3, 0.08]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
          <mesh position={[0.45, 1.1, 0]}>
            <boxGeometry args={[0.06, 0.25, 0.06]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
          <mesh position={[0.6, 0.9, 0]}>
            <boxGeometry args={[0.08, 0.05, 0.15]} />
            <meshStandardMaterial color="#fdbcb4" />
          </mesh>
        </group>
        
        {/* Legs */}
        <mesh position={[-0.1, 0.5, 0]}>
          <boxGeometry args={[0.08, 0.4, 0.08]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>
        <mesh position={[0.1, 0.5, 0]}>
          <boxGeometry args={[0.08, 0.4, 0.08]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>
        
        {/* Feet */}
        <mesh position={[-0.1, 0.05, 0.05]}>
          <boxGeometry args={[0.08, 0.05, 0.15]} />
          <meshStandardMaterial color="#34495e" />
        </mesh>
        <mesh position={[0.1, 0.05, 0.05]}>
          <boxGeometry args={[0.08, 0.05, 0.15]} />
          <meshStandardMaterial color="#34495e" />
        </mesh>
      </skinnedMesh>
      
      {/* Debug information */}
      {process.env.NODE_ENV === 'development' && (
        <mesh position={[0, 2.2, 0]}>
          <planeGeometry args={[1, 0.2]} />
          <meshBasicMaterial color="white" transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  );
};

export default Avatar3D;