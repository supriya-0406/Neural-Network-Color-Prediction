# Neural Network Color Prediction for Google Colab
# Install required packages first

!pip install tensorflow matplotlib seaborn scikit-learn numpy pandas plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
import random
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class ColorPredictionSystem:
    def __init__(self):
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.model = None
        self.history = None
        
    def rgb_to_hsv(self, r, g, b):
        """Convert RGB to HSV color space"""
        r, g, b = r/255.0, g/255.0, b/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h*360, s*100, v*100
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color space"""
        h, s, v = h/360.0, s/100.0, v/100.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r*255), int(g*255), int(b*255)
    
    def generate_color_dataset(self, n_samples=10000):
        """Generate synthetic color dataset with relationships"""
        print(f"Generating {n_samples} color samples...")
        
        data = []
        for _ in range(n_samples):
            # Generate random base color
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            
            # Convert to HSV for easier manipulation
            h, s, v = self.rgb_to_hsv(r, g, b)
            
            # Generate complementary color (opposite hue)
            comp_h = (h + 180) % 360
            comp_r, comp_g, comp_b = self.hsv_to_rgb(comp_h, s, v)
            
            # Generate analogous colors (±30 degrees)
            analog1_h = (h + 30) % 360
            analog2_h = (h - 30) % 360
            analog1_r, analog1_g, analog1_b = self.hsv_to_rgb(analog1_h, s, v)
            analog2_r, analog2_g, analog2_b = self.hsv_to_rgb(analog2_h, s, v)
            
            # Generate triadic colors (±120 degrees)
            triadic1_h = (h + 120) % 360
            triadic2_h = (h + 240) % 360
            triadic1_r, triadic1_g, triadic1_b = self.hsv_to_rgb(triadic1_h, s, v)
            triadic2_r, triadic2_g, triadic2_b = self.hsv_to_rgb(triadic2_h, s, v)
            
            # Calculate brightness and contrast
            brightness = (r + g + b) / 3
            contrast = max(r, g, b) - min(r, g, b)
            
            data.append({
                'input_r': r, 'input_g': g, 'input_b': b,
                'input_h': h, 'input_s': s, 'input_v': v,
                'brightness': brightness, 'contrast': contrast,
                'comp_r': comp_r, 'comp_g': comp_g, 'comp_b': comp_b,
                'analog1_r': analog1_r, 'analog1_g': analog1_g, 'analog1_b': analog1_b,
                'analog2_r': analog2_r, 'analog2_g': analog2_g, 'analog2_b': analog2_b,
                'triadic1_r': triadic1_r, 'triadic1_g': triadic1_g, 'triadic1_b': triadic1_b,
                'triadic2_r': triadic2_r, 'triadic2_g': triadic2_g, 'triadic2_b': triadic2_b
            })
        
        return pd.DataFrame(data)
    
    def build_neural_network(self, input_dim, output_dim):
        """Build neural network for color prediction"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df, target_type='complementary'):
        """Train the neural network"""
        print(f"Training model for {target_type} color prediction...")
        
        # Prepare features
        feature_cols = ['input_r', 'input_g', 'input_b', 'input_h', 'input_s', 'input_v', 'brightness', 'contrast']
        X = df[feature_cols].values
        
        # Prepare targets based on type
        if target_type == 'complementary':
            y = df[['comp_r', 'comp_g', 'comp_b']].values
        elif target_type == 'analogous':
            y = df[['analog1_r', 'analog1_g', 'analog1_b', 'analog2_r', 'analog2_g', 'analog2_b']].values
        elif target_type == 'triadic':
            y = df[['triadic1_r', 'triadic1_g', 'triadic1_b', 'triadic2_r', 'triadic2_g', 'triadic2_b']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_input.fit_transform(X_train)
        X_test_scaled = self.scaler_input.transform(X_test)
        
        # Scale targets
        y_train_scaled = self.scaler_output.fit_transform(y_train)
        y_test_scaled = self.scaler_output.transform(y_test)
        
        # Build and train model
        self.model = self.build_neural_network(X_train_scaled.shape[1], y_train_scaled.shape[1])
        
        # Training callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_test_scaled, y_test_scaled),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return X_test, y_test, y_pred
    
    def predict_colors(self, r, g, b):
        """Predict color relationships for given RGB values"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Prepare input
        h, s, v = self.rgb_to_hsv(r, g, b)
        brightness = (r + g + b) / 3
        contrast = max(r, g, b) - min(r, g, b)
        
        X = np.array([[r, g, b, h, s, v, brightness, contrast]])
        X_scaled = self.scaler_input.transform(X)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled)
        
        # Clip values to valid RGB range
        y_pred = np.clip(y_pred, 0, 255).astype(int)
        
        return y_pred[0]
    
    def visualize_training(self):
        """Visualize training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_color_prediction(self, r, g, b):
        """Visualize color predictions"""
        predictions = self.predict_colors(r, g, b)
        if predictions is None:
            return
        
        # Create color swatches
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Color Predictions for RGB({r}, {g}, {b})', fontsize=16)
        
        # Original color
        axes[0, 0].add_patch(plt.Rectangle((0, 0), 1, 1, color=(r/255, g/255, b/255)))
        axes[0, 0].set_title(f'Original Color\nRGB({r}, {g}, {b})')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        
        # Predicted complementary
        comp_r, comp_g, comp_b = predictions[:3]
        axes[0, 1].add_patch(plt.Rectangle((0, 0), 1, 1, color=(comp_r/255, comp_g/255, comp_b/255)))
        axes[0, 1].set_title(f'Predicted Complementary\nRGB({comp_r}, {comp_g}, {comp_b})')
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis('off')
        
        # Color wheel visualization
        axes[1, 0].set_title('Color Relationships')
        theta = np.linspace(0, 2*np.pi, 100)
        axes[1, 0].plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
        
        # Plot original color position
        h, s, v = self.rgb_to_hsv(r, g, b)
        orig_angle = np.radians(h)
        axes[1, 0].plot(np.cos(orig_angle), np.sin(orig_angle), 'o', 
                       color=(r/255, g/255, b/255), markersize=15, label='Original')
        
        # Plot complementary color position
        comp_angle = np.radians((h + 180) % 360)
        axes[1, 0].plot(np.cos(comp_angle), np.sin(comp_angle), 's', 
                       color=(comp_r/255, comp_g/255, comp_b/255), markersize=15, label='Complementary')
        
        axes[1, 0].set_xlim(-1.2, 1.2)
        axes[1, 0].set_ylim(-1.2, 1.2)
        axes[1, 0].set_aspect('equal')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RGB comparison
        colors = ['Original', 'Predicted']
        rgb_values = [[r, g, b], [comp_r, comp_g, comp_b]]
        
        x = np.arange(len(colors))
        width = 0.25
        
        axes[1, 1].bar(x - width, [rgb[0] for rgb in rgb_values], width, label='Red', color='red', alpha=0.7)
        axes[1, 1].bar(x, [rgb[1] for rgb in rgb_values], width, label='Green', color='green', alpha=0.7)
        axes[1, 1].bar(x + width, [rgb[2] for rgb in rgb_values], width, label='Blue', color='blue', alpha=0.7)
        
        axes[1, 1].set_xlabel('Colors')
        axes[1, 1].set_ylabel('RGB Values')
        axes[1, 1].set_title('RGB Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(colors)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_color_explorer(self):
        """Create interactive color exploration interface"""
        print("🎨 Interactive Color Explorer")
        print("=" * 50)
        
        while True:
            try:
                print("\nEnter RGB values (0-255) or 'quit' to exit:")
                user_input = input("RGB values (e.g., 255,128,64): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                r, g, b = map(int, user_input.split(','))
                
                if not all(0 <= val <= 255 for val in [r, g, b]):
                    print("Please enter values between 0 and 255")
                    continue
                
                print(f"\n🔍 Analyzing color RGB({r}, {g}, {b})...")
                self.visualize_color_prediction(r, g, b)
                
                # Additional analysis
                h, s, v = self.rgb_to_hsv(r, g, b)
                brightness = (r + g + b) / 3
                
                print(f"\n📊 Color Analysis:")
                print(f"HSV: ({h:.1f}°, {s:.1f}%, {v:.1f}%)")
                print(f"Brightness: {brightness:.1f}")
                print(f"Dominant Channel: {'Red' if r == max(r,g,b) else 'Green' if g == max(r,g,b) else 'Blue'}")
                
            except ValueError:
                print("Invalid input! Please enter three numbers separated by commas.")
            except KeyboardInterrupt:
                break
        
        print("\nThanks for using the Color Explorer! 🎨")

# Initialize and run the system
def main():
    print("🧠 Neural Network Color Prediction System")
    print("=" * 50)
    
    # Initialize system
    color_system = ColorPredictionSystem()
    
    # Generate dataset
    df = color_system.generate_color_dataset(n_samples=5000)
    print(f"Dataset shape: {df.shape}")
    
    # Train model
    X_test, y_test, y_pred = color_system.train_model(df, target_type='complementary')
    
    # Visualize training
    color_system.visualize_training()
    
    # Test with sample colors
    test_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (128, 64, 192), # Purple
    ]
    
    print("\n🎨 Testing with sample colors:")
    for r, g, b in test_colors:
        print(f"\nTesting RGB({r}, {g}, {b}):")
        color_system.visualize_color_prediction(r, g, b)
    
    # Interactive explorer
    print("\n🚀 Starting Interactive Color Explorer...")
    color_system.interactive_color_explorer()

# Run the main function
if __name__ == "__main__":
    main()
