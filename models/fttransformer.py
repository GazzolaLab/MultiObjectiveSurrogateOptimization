import math

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
        head_dim,
        input_norm=True,
        output_block=False,
        residual_dropout=0.0,
        ffn_activation="gelu",
    ):
        super(TransformerBlock, self).__init__()
        self.output_block = output_block
        self.att_normalization = (
            tf.keras.layers.LayerNormalization() if input_norm else None
        )
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=attention_dropout,
            output_shape=embedding_dimension,
        )
        self.attention_output_dropout = tf.keras.layers.Dropout(residual_dropout)
        self.ffn_normalization = tf.keras.layers.LayerNormalization()
        self.ffn_dense_1 = tf.keras.layers.Dense(
            ff_dimension, activation=ffn_activation
        )
        self.ffn_dropout = tf.keras.layers.Dropout(ffn_dropout)
        self.ffn_dense_2 = tf.keras.layers.Dense(embedding_dimension)
        self.ffn_output_dropout = tf.keras.layers.Dropout(residual_dropout)

    def call(self, inputs, training=None):
        x = inputs
        residual = x
        if self.att_normalization is not None:
            x = self.att_normalization(x, training=training)
        attention_out = self.mha(x, x, training=training)
        attention_out = self.attention_output_dropout(attention_out, training=training)
        x = residual + attention_out

        residual = x
        x = self.ffn_normalization(x, training=training)
        x = self.ffn_dense_1(x)
        x = self.ffn_dropout(x, training=training)
        x = self.ffn_dense_2(x)
        x = self.ffn_output_dropout(x, training=training)
        x = residual + x

        if self.output_block:
            x = x[:, 0]

        return x


class FTTransformer(Model):
    def prepare_layers(
        self,
        n_blocks=3,
        embedding_dim_per_head=32,
        num_heads=4,
        ffn_ratio=2.0,
        use_input_layer_norm=False,
        block_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.05,
        ffn_activation="gelu",
        pooling="cls",
        hidden_mlp_units=None,
        parameter_group_size=1,
        pool_every_n=0,
        pool_stride=2,
        pool_type="avg",
    ):
        embedding_dimension = embedding_dim_per_head * num_heads

        # reduce attention size by limiting effective tokens
        if parameter_group_size <= 1 and self.num_parameters > 64:
            parameter_group_size = max(
                parameter_group_size,
                math.ceil(self.num_parameters / 64),
            )

        # scale down for large number of objectives
        total_targets = self.num_objectives + self.num_constraints
        if total_targets > 100:
            n_blocks = min(n_blocks, 2)
            embedding_dimension = min(embedding_dimension, 128)
            num_heads = min(num_heads, 4)
            block_dropout = max(block_dropout, 0.1)
            parameter_group_size = max(parameter_group_size, 2)
            if pool_every_n == 0:
                pool_every_n = 1
        if total_targets > 200:
            n_blocks = 2
            embedding_dimension = min(embedding_dimension, 96)
            num_heads = min(num_heads, 3)
            parameter_group_size = max(parameter_group_size, 4)
            pool_stride = max(pool_stride, 2)

        if pooling == "cls":
            pool_every_n = 0

        head_dim = max(8, embedding_dimension // num_heads)

        d_rsqrt = embedding_dimension**-0.5
        self.embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.num_parameters * embedding_dimension,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.RandomUniform(
                        minval=-d_rsqrt, maxval=d_rsqrt
                    ),
                    bias_initializer=tf.keras.initializers.RandomUniform(
                        minval=-d_rsqrt, maxval=d_rsqrt
                    ),
                ),
                tf.keras.layers.Reshape((self.num_parameters, embedding_dimension)),
            ]
        )

        ffn_hidden = int(ffn_ratio * embedding_dimension)

        self.blocks = []
        self.token_poolers = []
        for block_index in range(n_blocks):
            block = TransformerBlock(
                embedding_dimension=embedding_dimension,
                ff_dimension=ffn_hidden,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                head_dim=head_dim,
                input_norm=use_input_layer_norm or block_index != 0,
                output_block=pooling == "cls" and block_index == n_blocks - 1,
                residual_dropout=block_dropout,
                ffn_activation=ffn_activation,
            )
            self.blocks.append(block)

            if (
                pooling != "cls"
                and pool_every_n
                and (block_index + 1) % pool_every_n == 0
            ):
                if pool_type == "avg":
                    pool_layer = tf.keras.layers.AveragePooling1D(
                        pool_size=pool_stride,
                        strides=pool_stride,
                        padding="same",
                    )
                elif pool_type == "max":
                    pool_layer = tf.keras.layers.MaxPooling1D(
                        pool_size=pool_stride,
                        strides=pool_stride,
                        padding="same",
                    )
                else:
                    raise ValueError(f"Unsupported pool_type: {pool_type}")
            else:
                pool_layer = None
            self.token_poolers.append(pool_layer)

        self.output_layer_norm = tf.keras.layers.LayerNormalization()

        if pooling == "mean":
            self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        elif pooling == "flatten":
            self.pooling_layer = tf.keras.layers.Flatten()
        else:
            self.pooling_layer = None

        hidden_mlp_units = list(hidden_mlp_units or [])
        self.head_mlp = tf.keras.Sequential(
            [
                layer
                for units in hidden_mlp_units
                for layer in (
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dropout(block_dropout),
                )
            ]
        )

        self.parameter_group_size = max(1, int(parameter_group_size))
        self.pooling_strategy = pooling

        self.architecture.update(
            {
                "n_blocks": n_blocks,
                "embedding_dim_per_head": embedding_dim_per_head,
                "num_heads": num_heads,
                "ffn_ratio": ffn_ratio,
                "use_input_layer_norm": use_input_layer_norm,
                "block_dropout": block_dropout,
                "attention_dropout": attention_dropout,
                "ffn_dropout": ffn_dropout,
                "ffn_activation": ffn_activation,
                "pooling": pooling,
                "hidden_mlp_units": hidden_mlp_units,
                "parameter_group_size": self.parameter_group_size,
                "pool_every_n": pool_every_n,
                "pool_stride": pool_stride,
                "pool_type": pool_type,
            }
        )

        self.objectives_output = tf.keras.layers.Dense(self.num_objectives)

        self.constraints_output = tf.keras.layers.Dense(
            self.num_constraints, activation="sigmoid", name="constraints"
        )

    def call(self, inputs, training=None):
        x = self.input_norm_layer(inputs)

        # x = self.norm(x)

        x = self.embedding(x)

        if self.parameter_group_size > 1:
            group_size = self.parameter_group_size
            x_shape = tf.shape(x)
            num_tokens = x_shape[1]
            remainder = tf.math.floormod(num_tokens, group_size)
            pad = tf.math.floormod(group_size - remainder, group_size)
            padding = tf.zeros((x_shape[0], pad, x_shape[2]), dtype=x.dtype)
            x = tf.concat([x, padding], axis=1)
            x_shape = tf.shape(x)
            new_tokens = x_shape[1] // group_size
            new_shape = tf.stack([x_shape[0], new_tokens, group_size, x_shape[2]])
            x = tf.reshape(x, new_shape)
            x = tf.reduce_mean(x, axis=2)

        for block, pooler in zip(self.blocks, self.token_poolers):
            x = block(x, training=training)
            if pooler is not None:
                x = pooler(x)

        x = self.output_layer_norm(x, training=training)

        if self.pooling_layer is not None:
            x = self.pooling_layer(x)

        if self.head_mlp.layers:
            x = self.head_mlp(x, training=training)

        if self.mode == "c+o":
            return {
                "objectives": self.objectives_output(x),
                "constraints": self.constraints_output(x),
            }

        if self.mode == "c":
            return self.constraints_output(x)

        if self.mode == "o":
            return self.objectives_output(x)
