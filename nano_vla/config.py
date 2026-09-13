"""Every tunable constant in one place. Change here, not in the modules."""

# --- temporal layout of one training window -------------------------------
IMG_HZ = 5.0  # image frames per second kept (30 fps source -> every 6th frame)
JOINT_HZ = 10.0  # joint rows per second kept (every 3rd frame)
WINDOW_S = 1.0  # seconds of past context the model sees
HORIZON_S = 1.0  # seconds of future joints the model predicts
T_IMG = int(WINDOW_S * IMG_HZ)  # 5 image frames per camera
T_JOINT = int(WINDOW_S * JOINT_HZ)  # 10 joint rows
HORIZON = int(HORIZON_S * JOINT_HZ)  # 10 future joint rows
JOINT_DIM = 6
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_IDX = 5

# --- trimming to the grasp segment -----------------------------------------
KEEP_AFTER_CLOSE_S = 1.5  # stored per episode (covers the lift)
LEVEL_AFTER_CLOSE_S = {"A": 0.5, "B": 1.5}  # what the loader keeps per level

# --- image representations ---------------------------------------------------
FRAME_RES = 160  # raw frames stored for the pixel MAE
DINO_MODEL = "dinov2_vits14"
DINO_RES = 224  # 224 / 14 = 16x16 patches
DINO_FEAT_DIM = 384
DINO_POOL = 2  # 16x16 -> 8x8 = 64 patches stored
DINO_N_PATCHES = (DINO_RES // 14 // DINO_POOL) ** 2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

# --- model defaults -----------------------------------------------------------
D_MODEL = 256
DEPTH = 4
HEADS = 4
DEC_D = 128
DEC_DEPTH = 2
