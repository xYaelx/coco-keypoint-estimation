import os


def create_checkpoint_dir(checkpoint_dir):
    """Create checkpoint directory if it doesn't exist"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f'Created checkpoint directory: {checkpoint_dir}')
