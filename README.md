# AI-Powered Foundation Shade Recommender

A computer vision and machine learning system that detects your skin tone in 
real time via webcam and recommends the closest matching foundation shade from 
a dataset of 1,000+ makeup products across multiple brands.

**Published:** D. M. Ruiz, A. Watson, Y. Kumar, J. J. Li & P. Morreale, 
"An AI-Powered Digital Foundation Recommender System," ISNCC 2024, 
Washington DC, USA, 2024, pp. 1-7.

---

## How It Works

1. **Webcam capture** — Opens a live video feed and draws a detection 
rectangle in the center of the frame
2. **Skin tone detection** — Extracts the region inside the rectangle, 
converts it to HSV color space, and isolates skin-colored pixels using 
color thresholding
3. **Color matching** — Computes the mean RGB value of detected skin pixels 
and finds the nearest cluster center using K-Means (k=6) trained on the 
makeup shades dataset
4. **Shade recommendation** — Returns the closest matching foundation shade 
with brand name, product name, and hex color code

---

## Tech Stack

- **Python** — Core language
- **OpenCV** — Webcam capture, image processing, HSV conversion
- **Scikit-learn** — K-Means clustering
- **NumPy / Pandas** — Data manipulation and RGB extraction
- **Dataset** — [Makeup Shades Dataset (Kaggle)](https://www.kaggle.com/datasets/shivamb/makeup-shades-dataset/data)

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/dahanam/ApplyAI_FoundationRecommender.git
cd ApplyAI_FoundationRecommender
```

### 2. Install dependencies
```bash
pip install opencv-python scikit-learn numpy pandas
```

### 3. Download the dataset
Download `shades.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/shivamb/makeup-shades-dataset/data) 
and place it in the root of the project folder.

### 4. Run the recommender
```bash
python FoundationRecommender2.py
```

### 5. Using the app
- A webcam window will open with a green rectangle in the center
- Position your face so your skin is inside the rectangle
- Press **any key** to capture and analyze your skin tone
- Your recommended foundation shade will print in the terminal
- Press **q** to quit
