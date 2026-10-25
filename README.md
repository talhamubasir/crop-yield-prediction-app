# Crop Yield Prediction — Assignment 3

## Files required
- `app.py` — Streamlit user interface
- `crop_yield.pkl` — trained model exported by `crop_yield.ipynb`
- `requirements.txt` — Python packages required by the app

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload these three files to the root of a **public GitHub repository**.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository, branch `main`, and file path `app.py`.
4. Click **Deploy** and wait for the build to finish.
5. Open the public app URL and test a prediction.

The app derives `log_rainfall`, `log_pesticides`, `temp_x_rainfall`, and `temp_squared` in the same way indicated by the Assignment 3 sample row. The notebook's saved model expects 18 features in a specific order.
