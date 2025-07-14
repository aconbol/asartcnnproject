import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from data.image_preprocessing import ImagePreprocessor

class RetinaDataset:
    def __init__(self, csv_path, batch_size=32, target_size=(256, 256)):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.filepaths = self.df['filepath'].values
        self.labels = self.df['level'].values
        
        self.current_index = 0
        
        # OPTIMIZACIÓN: Inicializar caché de imágenes para acelerar acceso
        self._image_cache = {}
        self._cache_size = 0
        self._max_cache_size = 50  # Ajustable según memoria disponible
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, filepath):
        """Load and preprocess single image"""
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Normalize to [0, 1]
        image = self.preprocessor.normalize_image(image)
        
        # Convert to channels-first format (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def get_batch(self, augment=False):
        """Obtener el siguiente lote de datos"""
        if self.current_index >= len(self.filepaths):
            # No reiniciar ni barajar automáticamente, solo devolver None
            return None, None
        
        end_index = min(self.current_index + self.batch_size, len(self.filepaths))
        batch_filepaths = self.filepaths[self.current_index:end_index]
        batch_labels = self.labels[self.current_index:end_index]
        
        # Cargar imágenes
        images = []
        labels = []
        
        for filepath, label in zip(batch_filepaths, batch_labels):
            try:
                image = self.load_image(filepath)
                
                if augment:
                    # Aplicar aumento (convertir a HWC para aumentar)
                    image_hwc = np.transpose(image, (1, 2, 0))
                    image_hwc = (image_hwc * 255).astype(np.uint8)
                    image_hwc, label = self.preprocessor.augment_image(image_hwc, label)
                    image = self.preprocessor.normalize_image(image_hwc)
                    image = np.transpose(image, (2, 0, 1))
                
                images.append(image)
                labels.append(label)
                
            except Exception as e:
                print(f"Error cargando {filepath}: {e}")
                continue
        
        self.current_index = end_index
        
        if len(images) == 0:
            return None, None
        
        return np.array(images), np.array(labels)
    
    def get_all_data(self):
        """Get all data at once"""
        images = []
        labels = []
        
        for filepath, label in zip(self.filepaths, self.labels):
            try:
                image = self.load_image(filepath)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def reset(self):
        """Reset iterator"""
        self.current_index = 0

class DataBalancer:
    """
    Data balancing utility
    Based on original balanze.py
    """
    
    @staticmethod
    def balance_dataset(csv_path, output_path):
        """Balance dataset classes"""
        df = pd.read_csv(csv_path)
        df['level'] = df['level'].astype(int)
        
        # Balance classes
        dfs = []
        max_size = int(df['level'].value_counts().values[0])
        
        for level in range(5):
            level = int(level)
            df_level = df[df['level'] == level]
            if len(df_level) < max_size:
                df_level_upsampled = resample(
                    df_level, 
                    replace=True, 
                    n_samples=max_size, 
                    random_state=42
                )
                dfs.append(df_level_upsampled)
            else:
                dfs.append(df_level)
        
        df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        df_balanced.to_csv(output_path, index=False)
        
        return df_balanced
    
    @staticmethod
    def create_small_dataset(csv_path, output_path, images_dir=None, image_extension='.jpeg', percentage=0.1, balance=True):
        """
        Create a smaller dataset by taking a percentage of each class
        
        Args:
            csv_path: Input CSV file path
            output_path: Output CSV file path
            images_dir: Directory containing images (optional)
            image_extension: Extension for image files (default: '.jpeg')
            percentage: Percentage of data to keep (0.1 = 10%)
            balance: Whether to balance classes after sampling
        """
        import os
        
        df = pd.read_csv(csv_path)
        df['level'] = df['level'].astype(int)
        
        # Add filepath column if it doesn't exist
        if 'filepath' not in df.columns and 'image' in df.columns:
            # Use provided images_dir or default to /mnt/d/Dataset_ret/dataset/train/
            base_path = images_dir or "/mnt/d/Dataset_ret/dataset/train/"
            df['filepath'] = df['image'].apply(lambda x: os.path.join(base_path, f"{x}{image_extension}"))
        
        # Take percentage of each class
        small_dfs = []
        for level in range(5):
            level = int(level)
            df_level = df[df['level'] == level]
            n_samples = max(1, int(len(df_level) * percentage))
            df_level_sample = df_level.sample(n=n_samples, random_state=42)
            small_dfs.append(df_level_sample)
        
        df_small = pd.concat(small_dfs).reset_index(drop=True)
        
        if balance:
            # Balance the small dataset
            dfs_balanced = []
            df_small['level'] = df_small['level'].astype(int)
            max_size = int(df_small['level'].value_counts().values[0])
            for level in range(5):
                level = int(level)
                df_level = df_small[df_small['level'] == level]
                if len(df_level) < max_size:
                    df_level_upsampled = resample(
                        df_level, 
                        replace=True, 
                        n_samples=max_size, 
                        random_state=42
                    )
                    dfs_balanced.append(df_level_upsampled)
                else:
                    dfs_balanced.append(df_level)
            
            df_small = pd.concat(dfs_balanced).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Asegúrate de que df_small es un DataFrame antes de guardar
        if isinstance(df_small, list):
            df_small = pd.concat(df_small).reset_index(drop=True)
        df_small.to_csv(output_path, index=False)
        print(f"🎯 Dataset pequeño creado: {len(df_small)} muestras ({percentage*100}% del original)")
        
        return df_small
    
    @staticmethod
    def split_dataset(csv_path, train_path, val_path, test_size=0.2, random_state=42):
        """
        Split dataset into train/validation
        Based on original split_csv.py
        
        Args:
            csv_path: Path to input CSV file
            train_path: Path for training CSV output
            val_path: Path for validation CSV output
            test_size: Fraction for validation (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        df = pd.read_csv(csv_path)
        
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['level'],
            random_state=random_state
        )
        
        # Asegúrate de que train_df y val_df son DataFrames antes de guardar
        if isinstance(train_df, list):
            train_df = pd.concat(train_df).reset_index(drop=True)
        if isinstance(val_df, list):
            val_df = pd.concat(val_df).reset_index(drop=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"Training: {len(train_df)} images")
        print(f"Validation: {len(val_df)} images")
        
        return train_df, val_df