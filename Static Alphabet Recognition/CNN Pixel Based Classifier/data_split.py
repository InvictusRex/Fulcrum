import os
import shutil
import random

DATASET_DIR = r'E:\1_Work_Files\13_Project - S1gnvers3\S1gnvers3\dataset\Gesture Image Data'
OUTPUT_DIR = r'E:\1_Work_Files\13_Project - S1gnvers3\S1gnvers3\dataset\Dataset_Split'

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, split_images in splits.items():
            split_cls_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_cls_dir, img)
                shutil.copyfile(src, dst)
        print(f'Split done for class {cls}: Train={len(splits["train"])}, Val={len(splits["val"])}, Test={len(splits["test"])}')

if __name__ == "__main__":
    split_dataset()