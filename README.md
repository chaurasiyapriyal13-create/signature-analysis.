# Signature Analysis Prototype

Upload two cropped signature images on white backgrounds to compare their visual structure.

## Run on Windows

Extract the ZIP and open the signature-analysis folder in VS Code. In its terminal run:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app opens in your browser. Upload both signatures and click Analyze Signatures.

## Method and limits

The app converts images to grayscale, thresholds the ink, crops and normalizes its size, and measures bidirectional stroke coverage with a small dilation tolerance. The score is visual overlap, not a probability of authenticity. The default threshold of 75 is unvalidated. Blank images are rejected.

This is an image-processing baseline, not a trained deep learning model. A trained CNN or Siamese network and appropriate evaluation data are still needed for the Deep Learning component of Practical 13.
