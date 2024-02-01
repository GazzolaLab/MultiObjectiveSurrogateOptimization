import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def apply_bounds(tensor, bounds):
    return tf.stack(
        [
            tf.clip_by_value(tensor[:, i], bounds[i][0], bounds[i][1])
            for i in range(len(bounds))
        ],
        axis=1,
    )


class MLP(tf.keras.models.Sequential):
    def __init__(
        self,
        num_parameters,
        num_constraints,
        num_objectives,
        learning_rate=0.1,
    ):
        super().__init__()
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives

        self.normalization_layer = tf.keras.layers.Normalization(
            input_shape=(num_parameters,)
        )

        self.add(self.normalization_layer)
        self.add(tf.keras.layers.Dense(100, activation="relu"))
        self.add(tf.keras.layers.Dense(num_constraints, activation="sigmoid"))

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                self.global_accuracy,
            ],
        )

        self.default_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=50,
                restore_best_weights=True,
                monitor="val_global_accuracy",
            )
        ]

        self.inverse_model = tf.keras.models.Model(
            inputs=self.input, outputs=self.output
        )

        self.inverse_input_sample = tf.Variable(
            initial_value=np.zeros([1, self.num_parameters]),
            dtype=tf.float32,
            name="inverse_input",
        )

    def make_feasible(
        self, X, learning_rate=0.1, transform="square", max_iterations=100, verbose=0
    ):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        for layer in self.layers:
            layer.trainable = False

        input_sample = tf.Variable(
            initial_value=X,
            dtype=tf.float32,
            name="inverse_X",
        )
        steps = tf.Variable(
            initial_value=tf.ones([X.shape[0]], dtype=np.int32) * -1,
            dtype=tf.int32,
            name="steps",
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy()

        iteration = 0
        while True:
            with tf.GradientTape() as tape:
                tape.watch(input_sample)

                # reparametrize to ensure positivity
                if isinstance(transform, (list, tuple)):
                    z = input_sample
                elif transform == "square":
                    z = tf.square(input_sample)
                elif transform == "exp":
                    z = tf.exp(input_sample)
                elif transform == "piece_exp":
                    z = tf.where(
                        input_sample > 0,
                        input_sample + 1,
                        tf.exp(input_sample),
                    )
                else:
                    raise ValueError(f"Invalid transform! {transform}")

                prediction = self(z)
                loss = loss_fn(
                    tf.constant(
                        np.ones([input_sample.shape[0], self.num_constraints]),
                        dtype=tf.float32,
                    ),
                    prediction,
                )

            if iteration > max_iterations:
                break

            grads = tape.gradient(loss, input_sample)

            is_feasible = tf.math.reduce_all(prediction > 0.99, axis=1)

            # record number of steps for feasible samples
            steps = tf.where(is_feasible, steps, iteration)

            # zero out grads for samples that are feasible
            is_feasible_where = tf.tile(
                tf.expand_dims(is_feasible, axis=1), [1, grads.shape[1]]
            )
            grads = tf.where(is_feasible_where, tf.zeros_like(grads), grads)

            optimizer.apply_gradients([(grads, input_sample)])

            if isinstance(transform, (list, tuple)):
                input_sample.assign(apply_bounds(input_sample, transform))

            iteration += 1

        for layer in self.layers:
            layer.trainable = True

        if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):
            # invalid optimization
            return X, False

        # inverse-transform
        if transform == "square":
            zp = np.square(input_sample.numpy())
        elif transform == "exp":
            zp = np.exp(input_sample.numpy())
        elif transform == "piece_exp":
            zp = np.where(
                input_sample.numpy() > 0,
                input_sample.numpy() + 1,
                np.exp(input_sample.numpy()),
            )
        else:
            zp = input_sample.numpy()

        return zp, steps.numpy()

    def global_accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(tf.cast(tf.reduce_all(y_true, axis=1), tf.int32), tf.float32)
        y_pred = tf.cast(tf.cast(tf.reduce_all(y_pred, axis=1), tf.int32), tf.float32)
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)

    def interactive(self):
        from livelossplot import PlotLossesKeras

        if len(self.default_callbacks) <= 1:
            self.default_callbacks.append(PlotLossesKeras())

    def fit(self, x=None, y=None, *args, callbacks=None, **kwargs):
        self.normalization_layer.adapt(x)
        if callbacks is None:
            callbacks = self.default_callbacks
        super().fit(x, y, *args, callbacks=callbacks, **kwargs)

    def eval(self, X_test, y_test, per_feature=False):
        y_prob = self.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)

        y_test_prime = y_test.all(axis=1).astype(int)
        y_pred_prime = y_pred.all(axis=1).astype(int)

        if per_feature:
            tbl = [["Constraint", "Precision", "Recall", "F1"]]
            labels = per_feature
            prec = precision_score(y_test, y_pred, average=None)
            rec = recall_score(y_test, y_pred, average=None)
            f1 = f1_score(y_test, y_pred, average=None)
            for t in zip(labels, prec, rec, f1):
                tbl.append(t)
            tbl.append(
                [
                    "Total",
                    precision_score(y_test_prime, y_pred_prime),
                    recall_score(y_test_prime, y_pred_prime),
                    f1_score(y_test_prime, y_pred_prime),
                ]
            )
            return tbl

        return {
            "accuracy": accuracy_score(y_test_prime, y_pred_prime),
            "precision": precision_score(y_test_prime, y_pred_prime),
            "recall": recall_score(y_test_prime, y_pred_prime),
            "f1": f1_score(y_test_prime, y_pred_prime),
        }
