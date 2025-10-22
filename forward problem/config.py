# Data settings
DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"  # Update this path
TARGET_SIZE = 56
TRAIN_RATIO = 0.75

# Model settings
MODES1 = 12  # Number of Fourier modes in first dimension
MODES2 = 12  # Number of Fourier modes in second dimension  
WIDTH = 64   # Hidden dimension
IN_CHANNELS = 2   # material_mask + bc_disp
OUT_CHANNELS = 2  # ux + uy

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 75
WEIGHT_DECAY = 1e-4
STEP_SIZE = 50  # For learning rate scheduler
GAMMA = 0.5     # Learning rate decay factor

DEVICE = 'cpu'  # If i end up using a computer with a gpu, change to auto

# Checkpointing
SAVE_DIR = "checkpoints"
SAVE_EVERY = 10  # Save checkpoint every N epochs

# Visualization
PLOT_LOSSES = True
VIS_SAMPLES = 4  # Number of samples to visualize
