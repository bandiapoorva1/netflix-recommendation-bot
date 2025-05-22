import gradio as gr
from hybrid_recommender import get_hybrid_recommendations

def recommend(input_title):
    return "\n".join(get_hybrid_recommendations(input_title, user_id=1))

gr.Interface(fn=recommend, inputs="text", outputs="text", title="Netflix Recommender Bot").launch()
