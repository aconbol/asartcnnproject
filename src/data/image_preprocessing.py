import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
    def preprocess_retina_image(self, img_path):
        """
        Preprocess retina image with CLAHE and circular crop
        Based on original procesamiento.py
        """
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read: {img_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # CLAHE enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        image = cv2.merge([enhanced, enhanced, enhanced])
        
        # Circular crop
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        
        return image
    
    def normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image, label):
        """
        Apply data augmentation
        Based on original dataset.py augmentation
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            image = np.clip(127.5 + contrast * (image - 127.5), 0, 255).astype(np.uint8)
        
        return image, label
    
    def process_dataset(self, csv_input, output_dir, csv_output):
        """
        Process entire dataset
        Based on original procesamiento.py
        """
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(csv_input)
        new_filepaths = []
        
        print("Processing images...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            original_path = row['filepath']
            
            try:
                image = self.preprocess_retina_image(original_path)
                
                # Create new filename
                img_name = f"{os.path.basename(original_path).split('.')[0]}.png"
                save_path = os.path.join(output_dir, img_name)
                
                # Save as PNG
                cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                new_filepaths.append(save_path)
                
            except Exception as e:
                print(f"Error with {original_path}: {e}")
                new_filepaths.append("ERROR")
        
        # Create new CSV
        df['filepath'] = new_filepaths
        df_clean = df[df['filepath'] != "ERROR"]
        df_clean.to_csv(csv_output, index=False)
        
        print(f"Saved: {csv_output}")
        print(f"Total processed: {len(df_clean)} of {len(df)}")
        
        return df_clean