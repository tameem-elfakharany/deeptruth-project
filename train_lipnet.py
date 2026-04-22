import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LipNet-inspired Deepfake Detector')
    parser.add_argument('--base-dir', type=str, default=r"C:\Users\hp\Desktop\DeepTruth\data\raw\video",
                        help='Base directory containing the dataset')
    parser.add_argument('--model-dir', type=str, default=r"C:\Users\hp\Desktop\DeepTruth\models",
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--frames', type=int, default=20, help='Number of frames per video')
    args = parser.parse_args()

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Configuration
    BASE_DIR = args.base_dir
    REAL_DIR = os.path.join(BASE_DIR, "DFD_original_sequences")
    FAKE_DIR = os.path.join(BASE_DIR, "DFD_manipulated_sequences")
    MODEL_SAVE_DIR = args.model_dir
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    FRAMES_PER_VIDEO = args.frames
    FRAME_HEIGHT = 100
    FRAME_WIDTH = 100
    CHANNELS = 3
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = 1e-4

    print(f"\nConfiguration:")
    print(f"Base Dir: {BASE_DIR}")
    print(f"Model Dir: {MODEL_SAVE_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Frames: {FRAMES_PER_VIDEO}\n")

    # Data Preprocessing Functions
    def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO, resize=(FRAME_HEIGHT, FRAME_WIDTH)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return np.zeros((num_frames, resize[0], resize[1], 3), dtype=np.float32)
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            frame_indices = np.arange(total_frames)
            padding = num_frames - total_frames
            frame_indices = np.pad(frame_indices, (0, padding), 'edge')
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (resize[1], resize[0]))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
                
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
            
        return np.array(frames[:num_frames])

    def load_dataset_paths():
        real_videos = glob.glob(os.path.join(REAL_DIR, "**", "*.mp4"), recursive=True)
        if not real_videos:
            real_videos = glob.glob(os.path.join(REAL_DIR, "**", "*.avi"), recursive=True)
            
        fake_videos = glob.glob(os.path.join(FAKE_DIR, "**", "*.mp4"), recursive=True)
        if not fake_videos:
            fake_videos = glob.glob(os.path.join(FAKE_DIR, "**", "*.avi"), recursive=True)
            
        print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")
        
        if len(real_videos) == 0 and len(fake_videos) == 0:
            print("Error: No videos found. Please check your dataset paths.")
            sys.exit(1)
            
        filepaths = real_videos + fake_videos
        labels = [0] * len(real_videos) + [1] * len(fake_videos)
        
        return filepaths, labels

    def create_tf_dataset(filepaths, labels, batch_size=BATCH_SIZE, is_training=True):
        def generator():
            for path, label in zip(filepaths, labels):
                frames = extract_frames(path.decode('utf-8') if isinstance(path, bytes) else path)
                yield frames, label
                
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=100)
            
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    # Model Architecture
    def build_lipnet_3dcnn_model(input_shape=(FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)):
        inputs = keras.Input(shape=input_shape)
        
        x = layers.Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout3D(0.2)(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
        
        x = layers.Conv3D(filters=64, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout3D(0.2)(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
        
        x = layers.Conv3D(filters=96, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout3D(0.2)(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
        
        x = layers.GlobalAveragePooling3D()(x)
        
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs, outputs, name="LipNet_Deepfake_Detector")
        return model

    # Load Data
    print("Loading dataset paths...")
    filepaths, labels = load_dataset_paths()

    # Split data: 80% train, 10% validation, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(filepaths, labels, test_size=0.1, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=42, stratify=y_train_val)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    train_ds = create_tf_dataset(X_train, y_train, is_training=True)
    val_ds = create_tf_dataset(X_val, y_val, is_training=False)
    test_ds = create_tf_dataset(X_test, y_test, is_training=False)

    # Build and Compile Model
    print("\nBuilding model...")
    model = build_lipnet_3dcnn_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, "lipnet_deepfake_best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train Model
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_auc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_DIR, "lipnet_deepfake_final.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training plot
    plot_path = os.path.join(MODEL_SAVE_DIR, "training_history.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    main()
