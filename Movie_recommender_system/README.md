# RUN WITHOUT DOCKER

Train:

python training/train.py

Run backend:

uvicorn app.main:app --reload

Run frontend:

streamlit run frontend/streamlit_app.py