import tensorflow as tf
from tensorflow.keras.utils import plot_model
from utils import LearnableCLSTokens, SinusoidalPositionEncoding, LearnablePositionEncoding

custom_objects = {
    'LearnableCLSTokens': LearnableCLSTokens,
    'SinusoidalPositionEncoding': SinusoidalPositionEncoding,
    'LearnablePositionEncoding': LearnablePositionEncoding,
}

model = tf.keras.models.load_model(
    r'data/models/jaad/CMFT/06May2026-11h44m16s/model.h5',
    custom_objects=custom_objects,
    compile=False
)

plot_model(
    model,
    to_file='model_graph.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False
)

print("Saved to model_graph.png")