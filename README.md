# Netflix Recommendation Bot

A hybrid movie/TV recommender using BERT for descriptions and SVD for collaborative filtering.

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train collaborative model:
```bash
python collaborative/Train_Collaborative_Model.py
```

3. Train BERT embeddings:
```bash
python content/Train_BERT_Embeddings.py
```

4. Run chatbot:
```bash
python gradio_chatbot.py
```
