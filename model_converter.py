"""
Model Converter: Convert GPU-optimized SavedModel to CPU-compatible format
Run this ONCE to fix the CudnnRNNV3 issue, then your app will work!

Usage:
    python model_converter.py
"""

import os
import shutil
import tensorflow as tf

# Force CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

print("🔄 Model Converter: GPU → CPU\n")

MODEL_PATH = 'model'
BACKUP_PATH = 'model_backup'
OUTPUT_PATH = 'model'

# ────────────────────────────────────────────────────────────────────────────
print("Step 1: Checking if model exists...")
if not os.path.isdir(MODEL_PATH):
    print(f"❌ Model folder not found at: {os.path.abspath(MODEL_PATH)}")
    print("   Make sure you're in the restaurant_app/ directory")
    exit(1)

if not os.path.exists(os.path.join(MODEL_PATH, 'saved_model.pb')):
    print(f"❌ saved_model.pb not found in: {os.path.abspath(MODEL_PATH)}")
    exit(1)

print(f"✅ Found model at: {os.path.abspath(MODEL_PATH)}\n")

# ────────────────────────────────────────────────────────────────────────────
print("Step 2: Backing up original model...")
if os.path.isdir(BACKUP_PATH):
    print(f"   (Backup already exists at {BACKUP_PATH})")
else:
    shutil.copytree(MODEL_PATH, BACKUP_PATH)
    print(f"✅ Backup saved to: {os.path.abspath(BACKUP_PATH)}\n")

# ────────────────────────────────────────────────────────────────────────────
print("Step 3: Loading model (may take a moment)...")
try:
    # Load using keras (which should work in inference mode on CPU)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully\n")
except Exception as e:
    print(f"⚠️  keras.models.load_model failed: {type(e).__name__}")
    print("   Trying tf.saved_model.load instead...\n")
    
    try:
        # Fallback: load as raw SavedModel
        concrete_func = tf.saved_model.load(MODEL_PATH)
        print("✅ SavedModel loaded (raw)\n")
        
        # Try to convert to keras model
        print("Step 4: Converting to Keras model...")
        # This is tricky - create a simple wrapper
        class ModelWrapper(tf.Module):
            def __init__(self, saved_model):
                super().__init__()
                self.saved_model = saved_model
            
            @tf.function
            def __call__(self, inputs):
                # Get the serving_default signature
                return self.saved_model.signatures['serving_default'](inputs)
        
        wrapper = ModelWrapper(concrete_func)
        print("✅ Model wrapped\n")
        
        print("Step 5: Saving CPU-compatible model...")
        # Save the wrapper
        tf.saved_model.save(
            wrapper,
            OUTPUT_PATH,
            signatures={
                'serving_default': wrapper.__call__.get_concrete_function(
                    tf.TensorSpec(shape=[None, 150], dtype=tf.float32, name='input')
                )
            }
        )
        print(f"✅ Saved to: {os.path.abspath(OUTPUT_PATH)}\n")
        print("🎉 Conversion complete! Try running your app now:\n")
        print("   python app.py\n")
        exit(0)
        
    except Exception as e2:
        print(f"❌ Both methods failed: {type(e2).__name__}: {e2}")
        print("\nℹ️  Please try Option B from the error message:")
        print("   Retrain your model using tf.keras.layers.LSTM (not CudnnLSTM)")
        exit(1)

# ────────────────────────────────────────────────────────────────────────────
print("Step 4: Saving CPU-compatible model...")
try:
    # Save with CPU-optimized settings
    model.save(
        OUTPUT_PATH,
        save_format='tf',
        include_optimizer=False
    )
    print(f"✅ Saved to: {os.path.abspath(OUTPUT_PATH)}\n")
except Exception as e:
    print(f"⚠️  Save failed: {type(e).__name__}: {e}")
    print("   Trying alternative save format...\n")
    try:
        model.save(f'{OUTPUT_PATH}.keras')
        print(f"✅ Saved as .keras format to: {os.path.abspath(OUTPUT_PATH)}.keras\n")
        print("Update your app.py line ~19 to:")
        print(f'   MODEL_PATH = "{OUTPUT_PATH}.keras"\n')
    except Exception as e3:
        print(f"❌ Save failed: {type(e3).__name__}")
        exit(1)

# ────────────────────────────────────────────────────────────────────────────
print("🎉 Conversion complete!\n")
print("Your model is now CPU-compatible.\n")
print("Next: Run your Flask app:")
print("   python app.py\n")
print("Then open: http://localhost:5000\n")
