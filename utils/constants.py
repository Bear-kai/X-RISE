import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.5, -0.5, 0], dtype=np.float32), np.array([0.5, 0.5, 1.0], dtype=np.float32) 
MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0], dtype=np.float32)
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0], dtype=np.float32)

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0], dtype=np.float32)
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4], dtype=np.float32)

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter

# ----------------------- for robotwin -----------------------
# workspace in global coordinate
TRANS_MIN_robotwin = np.array([-0.6, -0.35, 0.7], dtype=np.float32)
TRANS_MAX_robotwin = np.array([0.6, 0.35, 2], dtype=np.float32)
