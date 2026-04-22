import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants matching the LipNet notebook
FRAMES_PER_VIDEO = 20
FRAME_HEIGHT = 100
FRAME_WIDTH = 100
CHANNELS = 3

def extract_video_frames(
    video_path: str, 
    num_frames: int = FRAMES_PER_VIDEO, 
    resize: tuple = (FRAME_HEIGHT, FRAME_WIDTH)
) -> Optional[np.ndarray]:
    """
    Extracts a fixed number of frames from a video evenly spaced throughout the duration.
    Matches the logic in train_lipnet_classifier_v2.ipynb
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.error(f"Video has 0 frames: {video_path}")
            cap.release()
            return None
        
        # Determine which frames to extract
        if total_frames < num_frames:
            # If fewer frames than requested, repeat the last frame
            frame_indices = np.arange(total_frames)
            padding = num_frames - total_frames
            frame_indices = np.pad(frame_indices, (0, padding), 'edge')
        else:
            # Evenly space the frames across the video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Preprocessing: BGR to RGB, Resize, Normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (resize[1], resize[0]))
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
            else:
                # If frame reading fails, use a zero frame as fallback
                frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
                
        cap.release()
        
        # Final safety check on frame count
        while len(frames) < num_frames:
            frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
            
        return np.array(frames[:num_frames])
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None
