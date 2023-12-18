import os
import numpy as np
import mlflow
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.image import grayscale_to_rgb
from dorothy_sdk.operation import ModelOperation
from utils.convnets import train_neural_net, create_cnn


######## INSTRUÇÕES DE EDIÇÃO ############################

# Abaixo segue o código de exemplo para a execução do dummy_classifier, que é um modelo desenvolvido em PyTorch.
# Você deve alterar o código abaixo para a execução específica do seu modelo, porém vc deve respeitar todas
# os requisitos de entrada e saída do método abaixo.
#
# Caso você precise alterar algo mais específico do seu modelo, faça-o através de herança e
# sobrecarga dos métodos da classe ModelOperation (operation_base.py), lembrando sempre de
# respeitar todas os requisitos de entrada e saída dos métodos da classe ModelOperation.
#


class ProductionModelOperation(ModelOperation):
    def __init__(self, device_type=os.environ.get("DEVICE_TYPE", "cpu")):
        """
        Aqui você precisa definir qual o flavor do MLFlow seu modelo está usando. No meu exemplo,
        é o mlflow.pytorch.
        """
        super().__init__(
            mlflow_module_flavor=mlflow.tensorflow,
            device_type=device_type,
            img_path_col_name="image_local_path",
            label_col_name="y_true",
        )
        self.image_shape = (256, 256)

    def get_device(self):
        if self.device_type == "cpu":
            return "/CPU:0"
        return "/GPU:0"

    def prepare_images(self, X: np.ndarray):
        X = np.expand_dims(X, axis=3)
        X = grayscale_to_rgb(tf.constant(X))
        return tf.image.convert_image_dtype(X, tf.float32)

    def operate_model(self, model, X, decision_threshold: float = 0.5):
        """
        Aqui você deve definir o código responsável por passar o conjunto de imagens pelo modelo.
        Esta função deve retornar 2 numpy.arrays:
            - y_proba: contendo o valor da saída do modelo para cada amostra em X.
            - y_pred: contendo a decisão do modelo para cada amostra em X

        VOCÊ DEVE ALTERAR ESTA FUNÇÃO PARA O SEU CASO ESPECÍFICO, ATENTANDO QUE ELA
        PRECISA MANTER INALTERADOS SEUS PARÂMETROS DE ENTRADA E SAÍDA.
        """

        with tf.device(self.device):
            X = self.prepare_images(X)
            y_proba = model.predict(
                X, batch_size=64, verbose=0, workers=10, use_multiprocessing=True
            ).flatten()
            y_pred = np.zeros(len(y_proba))
            y_pred[y_proba >= decision_threshold] = 1.0
            return y_proba, y_pred

    def predict(self, model, X, decision_threshold: float = 0.5):
        with tf.device(self.device):
            X = np.expand_dims(X, axis=0)
            X = self.prepare_images(X)
            return 1.0 if model(X) >= decision_threshold else 0.0

    def _fix_df(self, df):
        col_map = {
            self.img_path_col_name: "path",
            self.label_col_name: "label",
        }
        return df.rename(columns=col_map)

    def train_model(self, trn_df, val_df, **kwargs):
        kwargs["image_shape"] = self.image_shape
        weights = train_neural_net(
            self._fix_df(trn_df), self._fix_df(val_df), kwargs
        ).model_weights
        model = create_cnn(self.image_shape)
        model.set_weights(weights)
        return model

    def _save_model(self, model, file_name: str, threshold: float) -> None:
        model.save(file_name)
        thres_file_name = os.path.join(file_name, "threshold.pickle")
        with open(thres_file_name, "wb") as f:
            pickle.dump(threshold, f)

    def _load_model(self, file_name: str):
        model = threshold = None
        with tf.device(self.device):
            model = keras.models.load_model(file_name, compile=True)

        thres_file_name = os.path.join(file_name, "threshold.pickle")
        with open(thres_file_name, "rb") as f:
            threshold = pickle.load(f)
        return model, threshold
