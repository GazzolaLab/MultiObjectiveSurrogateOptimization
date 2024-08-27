import tensorflow as tf
from models.model import Model


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dimension,
        ff_dimension,
        num_heads,
        attention_dropout,
        ffn_dropout,
        input_norm=True,
        output_block=False,
    ):
        super(TransformerBlock, self).__init__()
        self.output_block = output_block
        self.att_normalization = None
        if input_norm:
            self.att_normalization = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=embedding_dimension,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.ffn_normalization = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dimension, activation="relu"),
                tf.keras.layers.Dropout(ffn_dropout),
                tf.keras.layers.Dense(embedding_dimension),
            ]
        )

    def call(self, inputs, training=None):
        x = inputs
        x_identity = x
        if self.att_normalization:
            x = self.att_normalization(x, training=training)
        x = (
            self.mha(x, x, training=training)
            if not self.output_block
            else self.mha(x[:, :1], x, training=training)
        )
        x = tf.keras.layers.Add()([x_identity, x])

        x_identity = x
        x = self.ffn_normalization(x, training=training)
        x = self.ffn(x, training=training)
        x = tf.keras.layers.Add()([x_identity, x])

        if self.output_block:
            x = x[:, 0]

        return x


class FTTransformer(Model):
    def prepare_layers(self):
        n_blocks = 3
        d_block = [96, 128, 192, 256, 320, 384][n_blocks - 1]
        attention_dropout = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1]
        ffn_d_hidden_multiplier = 2
        ffn_dropout = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1]

        d_rqsrt = d_block**-0.5
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.num_parameters * d_block,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.RandomUniform(
                        minval=-d_rqsrt, maxval=d_rqsrt
                    ),
                    bias_initializer=tf.keras.initializers.RandomUniform(
                        minval=-d_rqsrt, maxval=d_rqsrt
                    ),
                ),
                tf.keras.layers.Reshape((self.num_parameters, d_block)),
            ]
        )

        self.blocks = [
            TransformerBlock(
                embedding_dimension=d_block,
                ff_dimension=ffn_d_hidden_multiplier * d_block,
                num_heads=8,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                input_norm=i != 0,
                output_block=i == n_blocks - 1,
            )
            for i in range(n_blocks)
        ]

        self.output_layer_norm = tf.keras.layers.LayerNormalization()

        # self.flattener = tf.keras.layers.Flatten()

        # self.mlps = [
        #     tf.keras.layers.Dense(n, activation="relu")
        #     for n in [1024, 256, 128, 64, 32]
        # ]

        self.objectives_output = tf.keras.layers.Dense(self.num_objectives)

        self.constraints_output = tf.keras.layers.Dense(
            self.num_constraints, activation="sigmoid", name="constraints"
        )

    def call(self, inputs, training=None):
        x = self.input_norm_layer(inputs)

        # x = self.norm(x)

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.output_layer_norm(x)
        x = tf.keras.layers.ReLU()(x)

        # x = self.flattener(x)

        # for mlp in self.mlps:
        #     x = mlp(x)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }

        if self.mode == "c":
            return self.constraints_output(x)

        if self.mode == "o":
            return self.objectives_output(x)
