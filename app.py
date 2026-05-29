# ── Force CPU-only mode (fixes CudnnRNNV3 GPU operations on CPU) ────────────
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU, use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF warnings

from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import pickle
import tensorflow as tf

# ✅ CRITICAL: Explicitly disable GPU before any model loading
tf.config.set_visible_devices([], 'GPU')
print("✅ GPU disabled — using CPU only")

from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Constants — must match your training notebook ────────────────────────────
MAX_WORDS      = 20000
MAX_LEN        = 150
MODEL_PATH = 'model/restaurant_rating_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'

# ── Startup diagnostics ───────────────────────────────────────────────────────
print("⏳ Loading model and tokenizer...")
print(f"   Working directory : {os.getcwd()}")
print(f"   Looking for model : {os.path.abspath(MODEL_PATH)}")
print(f"   Model folder exists: {os.path.isdir(MODEL_PATH)}")

# Show what's actually inside the model folder (helps diagnose nesting issues)
if os.path.isdir(MODEL_PATH):
    contents = os.listdir(MODEL_PATH)
    print(f"   Contents: {contents}")
else:
    print("   ⚠️  Model folder NOT FOUND.")
    print("   Make sure you have:  restaurant_app/model/restaurant_rating_model/saved_model.pb")

# ── Model loading — tries three strategies in order ──────────────────────────
_model_loaded = False

# Strategy 1: keras.models.load_model  (works for .keras / .h5; fails on Keras 3 SavedModel)
if not _model_loaded:
    try:
        _model = tf.keras.models.load_model(MODEL_PATH)
        def _run_model(padded):
            return _model.predict(padded, verbose=0)[0]
        print("✅ Model loaded via keras.models.load_model")
        _model_loaded = True
    except Exception as e:
        print(f"   ℹ️  keras.models.load_model failed: {type(e).__name__}")

# Strategy 2: TFSMLayer with 'serving_default' endpoint (standard TF SavedModel signature)
if not _model_loaded:
    try:
        _tfsm = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
        def _run_model(padded):
            tensor = tf.constant(padded, dtype=tf.float32)
            result = _tfsm(tensor)
            probs = list(result.values())[0].numpy()[0] if isinstance(result, dict) else result.numpy()[0]
            return probs
        print("✅ Model loaded via TFSMLayer (endpoint=serving_default)")
        _model_loaded = True
    except Exception as e:
        error_msg = str(e)
        if 'CudnnRNNV3' in error_msg:
            print(f"   ⚠️  TFSMLayer failed: Model uses GPU-only LSTM (CudnnRNNV3) — need conversion")
        else:
            print(f"   ℹ️  TFSMLayer serving_default failed: {type(e).__name__}")

# Strategy 3: TFSMLayer with 'serve' endpoint (used by some Keras exports)
if not _model_loaded:
    try:
        _tfsm = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serve')
        def _run_model(padded):
            tensor = tf.constant(padded, dtype=tf.float32)
            result = _tfsm(tensor)
            probs = list(result.values())[0].numpy()[0] if isinstance(result, dict) else result.numpy()[0]
            return probs
        print("✅ Model loaded via TFSMLayer (endpoint=serve)")
        _model_loaded = True
    except Exception as e:
        error_msg = str(e)
        if 'CudnnRNNV3' in error_msg:
            print(f"   ⚠️  TFSMLayer failed: Model uses GPU-only LSTM (CudnnRNNV3) — need conversion")
        else:
            print(f"   ℹ️  TFSMLayer serve endpoint failed: {type(e).__name__}")

if not _model_loaded:
    raise RuntimeError(
        "\n\n❌ Could not load the model.\n\n"
        "Your model uses CudnnRNNV3 (GPU-only LSTM ops). To fix:\n\n"
        "OPTION A: Run model_converter.py (one-time conversion):\n"
        "   python model_converter.py\n\n"
        "OPTION B: Train/retrain your model with a CPU-compatible LSTM\n\n"
        "Otherwise, check:\n"
        f"   1. Folder exists: {os.path.abspath(MODEL_PATH)}\n"
        "   2. Contains saved_model.pb directly (not nested)\n"
        "   3. TensorFlow installed: pip install tensorflow\n"
    )

# ── Load tokenizer ────────────────────────────────────────────────────────────
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"\n\n❌ tokenizer.pkl not found at: {os.path.abspath(TOKENIZER_PATH)}\n"
        "   Copy tokenizer.pkl from Google Drive into the restaurant_app/ folder.\n"
    )

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer loaded — ready to serve!\n")


# ── Text cleaning — identical to the notebook ────────────────────────────────
def clean_review(text: str) -> str:
    text = str(text)
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9\s\'.!?,]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("'", '')
    return text.lower()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_rating(review_text: str):
    cleaned    = clean_review(review_text)
    seq        = tokenizer.texts_to_sequences([cleaned])
    padded     = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs      = _run_model(padded)
    pred_star  = int(np.argmax(probs)) + 1
    confidence = [round(float(p) * 100, 1) for p in probs]
    return pred_star, confidence


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json(silent=True) or {}
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'Please enter a review.'}), 400
    if len(review) < 10:
        return jsonify({'error': 'Review is too short — write at least a sentence.'}), 400

    try:
        rating, confidence = predict_rating(review)
        return jsonify({'rating': rating, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
