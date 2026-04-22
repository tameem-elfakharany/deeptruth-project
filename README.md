# DeepTruth: Image Deepfake Detection

This project is a local VS Code adaptation of the Google Colab image deepfake detection tool. It uses a pre-trained XceptionNet model to analyze images and provide deepfake probability scores along with the top-5 ImageNet predictions.

## Project Structure

```
DeepTruth/
├── data/
│   ├── raw/                 # Put your dataset here
│   │   ├── real/            # Real images
│   │   └── fake/            # Fake images
│   └── processed/
├── models/                  # Saved models will go here
├── notebooks/
│   └── image_deepfake_detection.ipynb  # Main Jupyter notebook
├── src/
│   └── api/
│       └── main.py          # FastAPI backend
└── requirements.txt         # Project dependencies
```

## Setup Instructions

### 1. Create Virtual Environment

Open a terminal in the project root directory:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 3. Prepare Dataset

Create the following folder structure and place your dataset (e.g., DFDC or Celeb-DF) inside:
- `data/raw/real/` - Place real images here
- `data/raw/fake/` - Place deepfake images here

## Usage

### Option 1: Jupyter Notebook (Training & Inference)

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `notebooks/image_deepfake_detection.ipynb`
3. Run the cells to:
   - Load the Xception model
   - Preprocess your images exactly like in Colab
   - Make predictions and visualize results
   - Save the model to `models/xception_deepfake.h5`

### Option 2: FastAPI Backend (Production Inference)

Once you've saved the model (or if you want to use the pre-trained ImageNet weights), you can run the API:

1. Start the FastAPI server:
   ```bash
   cd src/api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Open your browser and go to:
   - Interactive API docs: `http://localhost:8000/docs`
   - API Info: `http://localhost:8000/info`

3. Test the API using Python:
   ```python
   import requests
   
   with open('test_image.jpg', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/predict',
           files={'file': f}
       )
       print(response.json())
   ```

## Model Details

- **Architecture**: XceptionNet
- **Input Size**: 299x299 RGB
- **Preprocessing**: 
  - Resize to 299x299
  - Convert Grayscale to RGB (if needed)
  - Add batch dimension
  - Normalize using `preprocess_input`
- **Output**: Top-5 predictions with confidence scores
