# How to Run Locally

### Step 1:

Train model:

python src/train.py

### Step 2:

Run API:

uvicorn app.main:app --reload

Visit:

http://127.0.0.1:8000/docs


# Docker Commands

Build image:

docker build -t churn-ml-app .

Run container:

docker run -p 8000:8000 churn-ml-app



# Run Without Docker

### Train Model

python training/train.py

### Run Backend

uvicorn app.main:app --reload

### Run Frontend

streamlit run frontend/streamlit_app.py