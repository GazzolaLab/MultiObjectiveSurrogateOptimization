import tensorflow as tf
from models.model import Model


class Resnet(Model):
    def prepare_layers(self, **kwargs):
        multihead = False
        n_blocks = 2
        d_block = 192
        d_hidden = None
        d_hidden_multiplier = 2.0
        dropout1 = 0.15
        dropout2 = 0.0

        if d_hidden is None:
            if d_hidden_multiplier is None:
                raise ValueError(
                    "If d_hidden is None, then d_hidden_multiplier must not be None"
                )
            d_hidden = int(d_block * d_hidden_multiplier)
        else:
            if d_hidden_multiplier is not None:
                raise ValueError(
                    "If d_hidden is not None, then d_hidden_multiplier must be None"
                )

        self.input_projection = tf.keras.layers.Dense(d_block)

        self.blocks = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(d_hidden),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dropout(dropout1),
                    tf.keras.layers.Dense(d_block),
                    tf.keras.layers.Dropout(dropout2),
                ]
            )
            for _ in range(n_blocks)
        ]

        self.out_norm = tf.keras.layers.BatchNormalization()

        self.gradnorm_layers.append(self.blocks[-1])

        if "o" in self.mode:
            if multihead:
                self.heads = [
                    tf.keras.layers.Dense(
                        1, activation="linear", name=f"regression_{head}_out"
                    )
                    for head in range(self.num_objectives)
                ]

                def multihead_regression(x):
                    return tf.keras.layers.Concatenate()(
                        [head(x) for head in self.heads]
                    )

                self.objectives_output = multihead_regression
            else:
                self.objectives_output = tf.keras.layers.Dense(
                    self.num_objectives,
                    name="objectives",
                )

        if "c" in self.mode:
            self.constraints_output = tf.keras.layers.Dense(
                self.num_constraints, activation="sigmoid", name="constraints"
            )

    def call(self, inputs):
        x = self.input_norm_layer(inputs)

        x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)

        # x = self.out_norm(x)
        # x = tf.keras.layers.ReLU()(x)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }

        if self.mode == "c":
            return self.constraints_output(x)

        if self.mode == "o":
            return self.objectives_output(x)
