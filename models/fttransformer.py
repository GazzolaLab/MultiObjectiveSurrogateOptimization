import tensorflow as tf
from models.model import Model


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, ff_dimension, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dimension
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dimension, activation="relu"),
                tf.keras.layers.Dense(embedding_dimension),
            ]
        )
        self.norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(2)]
        self.dol = [tf.keras.layers.Dropout(dropout) for _ in range(2)]

    def call(self, inputs, training):
        attention = self.mha(inputs, inputs)
        attention = self.dol[0](attention, training=training)
        x = self.norm[0](inputs + attention)
        ffo = self.ffn(x)
        ffo = self.dol[1](ffo, training=training)
        return self.norm[1](x + ffo)


class FTTransformer(Model):
    def prepare_layers(self):
        self.norm = tf.keras.layers.LayerNormalization()
        self.embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=16)

        self.transformers = [
            TransformerBlock(embedding_dimension=16, ff_dimension=16, num_heads=8)
            for _ in range(6)
        ]

        self.flattener = tf.keras.layers.Flatten()

        self.mlps = [
            tf.keras.layers.Dense(n, activation="relu")
            for n in [1024, 256, 128, 64, 32]
        ]

        self.objectives_output = tf.keras.layers.Dense(self.num_objectives)

        self.constraints_output = tf.keras.layers.Dense(
            self.num_constraints, activation="sigmoid", name="constraints"
        )

    def call(self, inputs):
        x = self.input_norm_layer(inputs)

        x = self.norm(x)
        x = self.embedding(x)

        for h in self.transformers:
            x = h(x)

        x = self.flattener(x)

        for mlp in self.mlps:
            x = mlp(x)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }

        if self.mode == "c":
            return self.constraints_output(x)

        if self.mode == "o":
            return self.objectives_output(x)
