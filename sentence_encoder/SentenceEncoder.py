import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # required

MODEL_DIRECTORY = "universal-sentence-encoder-multilingual-3"

if tf.saved_model.contains_saved_model(MODEL_DIRECTORY):
    _model = tf.saved_model.load(MODEL_DIRECTORY)
else:
    print("no local model found. Please wait during download")
    _model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    tf.saved_model.save(_model, MODEL_DIRECTORY)
    print("model saved in directory %s" % MODEL_DIRECTORY)


def encode_sentence(strs: tf.Tensor) -> tf.Tensor:
    return _model(strs)
