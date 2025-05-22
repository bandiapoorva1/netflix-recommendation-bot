
# 🎬 Netflix Recommendation Bot

A hybrid AI-powered recommendation system that suggests Netflix movies and TV shows based on what you've watched. It combines **collaborative filtering** (user-based preferences) and **content-based filtering** (using BERT embeddings of show descriptions) to provide personalized recommendations.

<p align="center">
  <img src="https://img.shields.io/github/license/bandiapoorva1/netflix-recommendation-bot?style=flat-square" alt="License" />
  <img src="https://img.shields.io/github/languages/top/bandiapoorva1/netflix-recommendation-bot?style=flat-square" alt="Top Language" />
  <img src="https://img.shields.io/github/last-commit/bandiapoorva1/netflix-recommendation-bot?style=flat-square" alt="Last Commit" />
</p>

---

## 🚀 Features

- 📊 **Collaborative Filtering** (SVD) based on user ratings
- 🧠 **Content-Based Filtering** using BERT sentence embeddings of Netflix descriptions
- 🔁 **Hybrid Scoring**: Similar content + user interest prediction
- 💬 **Gradio Chatbot**: Enter a Netflix title and get smart recommendations
- 🌐 **Ready for Deployment** via Streamlit or Hugging Face Spaces

---

## 📸 Demo

```bash
python gradio_chatbot.py
```

Visit: `http://127.0.0.1:7860` in your browser.

---

## 🧱 Architecture

```text
          +--------------------+
          | Netflix Dataset    |
          | (Descriptions)     |
          +---------+----------+
                    |
                    v
     +-------------------------------+
     | BERT Embedding (MiniLM)       |  --->  Content Similarity
     +-------------------------------+

          +--------------------+
          | MovieLens Dataset  |
          | (Ratings)          |
          +---------+----------+
                    |
                    v
     +-------------------------------+
     | Collaborative Filtering (SVD) |  --->  Rating Prediction
     +-------------------------------+

               +-----------------------------+
               | Hybrid Recommender System   |
               +-----------------------------+
                        |
                        v
              Recommendations Output (Top N)
```

---

## 📦 Dataset Sources

- 🎬 **Netflix Titles** (Kaggle): [Link](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- 🎥 **MovieLens 100K**: [Link](https://grouplens.org/datasets/movielens/)

---

## 🧰 Tech Stack

- Python 3.10+
- `pandas`, `numpy`, `scikit-surprise`
- `sentence-transformers` (`MiniLM`)
- `scikit-learn`, `gradio`
- Compatible with macOS & Linux

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/bandiapoorva1/netflix-recommendation-bot.git
cd netflix-recommendation-bot

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Train the collaborative model
python collaborative/Train_Collaborative_Model.py

# 5. Generate BERT embeddings
python content/Train_BERT_Embeddings.py

# 6. Run the chatbot
python gradio_chatbot.py
```

---

## 📂 Folder Structure

```
netflix-recommendation-bot/
│
├── data/                         # Raw CSV datasets
├── collaborative/               # SVD training script
├── content/                     # BERT embedding training
├── hybrid_recommender.py        # Core hybrid logic
├── gradio_chatbot.py            # Chatbot UI
├── streamlit_app.py             # (Optional) Web UI
├── requirements.txt             # Dependencies
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Fork this repo
- Add features or improve the logic
- Open pull requests

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙋‍♀️ Author

Made with ❤️ by [Apoorva Bandi](https://github.com/bandiapoorva1)

---

## 💡 Future Enhancements

- Add search autocomplete for better UX
- Integrate Firebase or Supabase for real-time feedback loop
- Deploy on Streamlit Cloud or Hugging Face Spaces
