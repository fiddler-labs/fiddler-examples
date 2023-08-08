import pathlib
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import fiddler as fdl

# Name the output of your model here - this will need to match the model schema we define in the next notebook
OUTPUT_COL = ['sentiment']

# These are the names of the inputs of yout TensorFlow model
FEATURE_LABEL = 'sentence'

MODEL_ARTIFACT_PATH = 'saved_model'

TOKENIZER_PATH = 'tokenizer.pkl'

ATTRIBUTABLE_LAYER_NAMES = EMBEDDING_NAMES = ['embedding']

MAX_SEQ_LENGTH = 150


def _pad(seq):
    return pad_sequences(seq, MAX_SEQ_LENGTH, padding='post', truncating='post')


class FiddlerModel:
    def __init__(self):
        """Model deserialization and initialization goes here.  Any additional serialized preprocession
        transformations would be initialized as well - e.g. tokenizers, embedding lookups, etc.
        """
        self.model_dir = pathlib.Path(__file__).parent
        self.model = tf.keras.models.load_model(
            str(self.model_dir / MODEL_ARTIFACT_PATH)
        )

        # Construct sub-models (for each ATTRIBUTABLE_LAYER_NAME)
        # if not possible to attribute directly to the input (e.g. embeddings).
        self.att_sub_models = {
            att_layer: Model(
                self.model.inputs, outputs=self.model.get_layer(att_layer).output
            )
            for att_layer in ATTRIBUTABLE_LAYER_NAMES
        }

        with open(str(self.model_dir / TOKENIZER_PATH), 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.grad_model = self._define_model_grads()

    def get_settings(self):

        # from ig_flex_exec.py
        # DEFAULT_START_STEPS = 32
        # DEFAULT_MAX_STEPS = 2048
        # DEFAULT_MAX_ERROR_PCT = 1.0

        return {
            'ig_start_steps': 32,  # 32
            'ig_max_steps': 4096,  # 2048
            'ig_min_error_pct': 5.0,  # 1.0
        }

    def transform_to_attributable_input(self, input_df):
        """This method is called by the platform and is responsible for transforming the input dataframe
        to the upstream-most representation of model inputs that belongs to a continuous vector-space.
        For this example, the model inputs themselves meet this requirement.  For models with embedding
        layers (esp. NLP models) the first attributable layer is downstream of that.
        """
        transformed_input = self._transform_input(input_df)

        return {
            att_layer: att_sub_model.predict(transformed_input)
            for att_layer, att_sub_model in self.att_sub_models.items()
        }

    def get_ig_baseline(self, input_df):
        """This method is used to generate the baseline against which to compare the input.
        It accepts a pandas DataFrame object containing rows of raw feature vectors that
        need to be explained (in case e.g. the baseline must be sized according to the explain point).
        Must return a pandas DataFrame that can be consumed by the predict method described earlier.
        """
        baseline_df = input_df.copy()
        baseline_df[FEATURE_LABEL] = input_df[FEATURE_LABEL].apply(lambda x: '')

        return baseline_df

    def _transform_input(self, input_df):
        """Helper function that accepts a pandas DataFrame object containing rows of raw feature vectors.
        The output of this method can be any Python object. This function can also
        be used to deserialize complex data types stored in dataset columns (e.g. arrays, or images
        stored in a field in UTF-8 format).
        """
        sequences = self.tokenizer.texts_to_sequences(input_df[FEATURE_LABEL])
        sequences_matrix = sequence.pad_sequences(
            sequences, maxlen=MAX_SEQ_LENGTH, padding='post'
        )
        return sequences_matrix.tolist()

    def predict(self, input_df):
        """Basic predict wrapper.  Takes a DataFrame of input features and returns a DataFrame
        of predictions.
        """
        transformed_input = self._transform_input(input_df)
        pred = self.model.predict(transformed_input)
        return pd.DataFrame(pred, columns=OUTPUT_COL)

    def compute_gradients(self, attributable_input):
        """This method computes gradients of the model output wrt to the differentiable input.
        If there are embeddings, the attributable_input should be the output of the embedding
        layer. In the backend, this method receives the output of the transform_to_attributable_input()
        method. This must return an array of dictionaries, where each entry of the array is the attribution
        for an output. As in the example provided, in case of single output models, this is an array with
        single entry. For the dictionary, the key is the name of the input layer and the values are the
        attributions.
        """
        gradients_by_output = []
        attributable_input_tensor = {
            k: tf.identity(v) for k, v in attributable_input.items()
        }
        gradients_dic_tf = self._gradients_input(attributable_input_tensor)
        gradients_dic_numpy = dict(
            [key, np.asarray(value)] for key, value in gradients_dic_tf.items()
        )
        gradients_by_output.append(gradients_dic_numpy)
        return gradients_by_output

    def _gradients_input(self, x):
        """
        Function to Compute gradients.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = self.grad_model(x)

        grads = tape.gradient(preds, x)

        return grads

    def _define_model_grads(self):
        """
        Define a differentiable model, cut from the Embedding Layers.
        This will take as input what the transform_to_attributable_input function defined.
        """
        model = tf.keras.models.load_model(str(self.model_dir / 'saved_model'))

        for index, name in enumerate(EMBEDDING_NAMES):
            model.layers.remove(model.get_layer(name))
            model.layers[index]._batch_input_shape = (None, 150, 64)
            model.layers[index]._dtype = 'float32'
            model.layers[index]._name = name

        new_model = tf.keras.models.model_from_json(model.to_json())

        for layer in new_model.layers:
            try:
                layer.set_weights(self.model.get_layer(name=layer.name).get_weights())
            except:
                pass

        return new_model

    #  Here's a project_attributions that works for a different single text input model

    # input_df: explain_point df from raw feature space (model_info)
    # attributions: array[<output_dims>] of dict{tensor_names: }
    #     of array[tensor_dims...]
    # returns: dict{output_names: } of feature attributions described in
    #     GEM [generalized explanation markup].
    def project_attributions(self, input_df, attributions):
        explanations_by_output = {}

        for output_field_index, att in enumerate(attributions):
            segments = re.split(
                r'([ ' + self.tokenizer.filters + '])', input_df.iloc[0][FEATURE_LABEL]
            )

            unpadded_tokens = [
                self.tokenizer.texts_to_sequences([x])[0]
                for x in input_df[FEATURE_LABEL].values
            ]

            padded_tokens = _pad(unpadded_tokens)

            word_tokens = self.tokenizer.sequences_to_texts(
                [[x] for x in padded_tokens[0]]
            )

            # Note - summing over attributions in the embedding direction
            word_attributions = np.sum(att['embedding'][-len(word_tokens) :], axis=1)

            i = 0
            final_attributions = []
            final_segments = []
            for segment in segments:
                if segment is not '':  # dump empty tokens
                    final_segments.append(segment)
                    seg_low = segment.lower()
                    if len(word_tokens) > i and seg_low == word_tokens[i]:
                        final_attributions.append(word_attributions[i])
                        i += 1
                    else:
                        final_attributions.append(0)

            gem_text = fdl.gem.GEMText(
                feature_name=FEATURE_LABEL,
                text_segments=final_segments,
                text_attributions=final_attributions,
            )

            gem_container = fdl.gem.GEMContainer(contents=[gem_text])

            explanations_by_output[
                OUTPUT_COL[output_field_index]
            ] = gem_container.render()

        return explanations_by_output


def get_model():
    return FiddlerModel()
