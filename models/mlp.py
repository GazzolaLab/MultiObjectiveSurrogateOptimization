import tensorflow as tf
from models.model import Model
from keras.layers import LeakyReLU


class MLP(Model):
    def prepare_layers(self):
        multihead = False

        # Use LeakyReLU to prevent zeroing of forward pass
        #  in low-data regimes
        # -> https://github.com/keras-team/keras/issues/6447
        self.hidden = [
            tf.keras.layers.Dense(
                i,
                activation=LeakyReLU(),
                # kernel_constraint=tf.keras.constraints.MaxNorm(3),
                # kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                name=f"hidden{i}",
            )
            for i in [32, 16, 8]
        ]
        if multihead:
            self.heads = [
                [
                    tf.keras.layers.Dense(
                        32, activation=LeakyReLU(), name=f"regression_{head}_d1"
                    ),
                    tf.keras.layers.Dense(1, name=f"regression_{head}_out"),
                ]
                for head in range(self.num_objectives)
            ]

            def multihead_regression(x):
                return tf.keras.layers.Concatenate()(
                    [head[1](head[0](x)) for head in self.heads]
                )

            self.objectives_output = multihead_regression
        else:
            self.objectives_output = tf.keras.layers.Dense(
                self.num_objectives,
                name="objectives",
            )

        self.constraints_output = tf.keras.layers.Dense(
            self.num_constraints, activation="sigmoid", name="constraints"
        )

    def call(self, inputs, training):
        x = self.input_norm_layer(inputs)
        for h in self.hidden:
            x = h(x)
            x = tf.keras.layers.Dropout(0.3)(x, training=training)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }

        if self.mode == "c":
            return self.constraints_output(x)

        if self.mode == "o":
            return self.objectives_output(x)
