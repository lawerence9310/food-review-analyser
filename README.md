# 🍽️ Taste Rater — Flask Web App

A web app that predicts restaurant star ratings (1–5) from review text using the LSTM model trained in the companion Colab notebook.

---

## Project Structure

```
restaurant_app/
├── app.py                        ← Flask backend
├── requirements.txt
├── templates/
│   └── index.html                ← Frontend UI
├── model/
│   └── restaurant_rating_model/  ← SavedModel folder (you copy this in)
└── tokenizer.pkl                 ← Tokenizer (you copy this in)
```

---

## Step 1 — Get Your Model & Tokenizer from Colab

After training, your notebook saves two things to Google Drive:
- `restaurant_rating_model/` — the SavedModel folder
- `tokenizer.pkl` — the tokenizer

**Download them to your computer:**

```python
# Run this in your Colab notebook to zip the model folder
import shutil
shutil.make_archive('/content/drive/MyDrive/restaurant_rating_model', 'zip',
                    '/content/drive/MyDrive/restaurant_rating_model')
```

Then download both `restaurant_rating_model.zip` and `tokenizer.pkl` from Google Drive.

**Place them in the app folder:**

```
restaurant_app/
├── model/
│   └── restaurant_rating_model/   ← unzipped contents go here
└── tokenizer.pkl                  ← goes here
```

---

## Step 2 — Install Dependencies

```bash
cd restaurant_app
pip install -r requirements.txt
```

> **Note:** TensorFlow can take a few minutes to install. If you only have CPU, that's fine — predictions are fast.

---

## Step 3 — Run the App

```bash
python app.py
```

You should see:
```
⏳ Loading model and tokenizer...
✅ Model loaded
✅ Tokenizer loaded — ready to serve!
 * Running on http://127.0.0.1:5000
```

Open **http://localhost:5000** in your browser.

---

## API

The app exposes one endpoint used by the frontend:

| Method | Endpoint   | Body                    | Response                                     |
|--------|------------|-------------------------|----------------------------------------------|
| POST   | `/predict` | `{ "review": "..." }`   | `{ "rating": 4, "confidence": [2,5,10,65,18] }` |

You can also call it directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "The food was incredible and staff very friendly!"}'
```

---

## Troubleshooting

**Model not found error**
Make sure the folder structure is exactly `model/restaurant_rating_model/` (not `model/restaurant_rating_model/restaurant_rating_model/`).

**`TFSMLayer` import error**
Your TensorFlow version may be older. The app automatically falls back to `keras.models.load_model` — if both fail, try upgrading:
```bash
pip install --upgrade tensorflow
```

**Tokenizer error**
The `tokenizer.pkl` must be the exact file saved at the end of your training notebook. A different tokenizer will produce garbage predictions.
