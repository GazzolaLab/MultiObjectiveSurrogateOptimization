import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        self, X, learning_rate=0.1, transform="square", max_iterations=1e5, verbose=0
    ):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        for layer in self.layers:
            layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy()

        self.inverse_input_sample.assign(X)

        step = 0
        while True:
            with tf.GradientTape() as tape:
                tape.watch(self.inverse_input_sample)

                # reparametrize to ensure positivity
                if isinstance(transform, (list, tuple)):
                    z = self.inverse_input_sample
                elif transform == "square":
                    z = tf.square(self.inverse_input_sample)
                elif transform == "exp":
                    z = tf.exp(self.inverse_input_sample)
                elif transform == "piece_exp":
                    z = tf.where(
                        self.inverse_input_sample > 0,
                        self.inverse_input_sample + 1,
                        tf.exp(self.inverse_input_sample),
                    )
                else:
                    raise ValueError(f"Invalid transform! {transform}")

                prediction = self(z)
                loss = loss_fn(
                    tf.constant(np.ones([1, self.num_constraints]), dtype=tf.float32),
                    prediction,
                )

            if loss < 1e-10 or step > max_iterations:
                break

            grads = tape.gradient(loss, self.inverse_input_sample)
            optimizer.apply_gradients([(grads, self.inverse_input_sample)])

            if isinstance(transform, (list, tuple)):
                v = self.inverse_input_sample.numpy()
                min_max_vals = np.array(transform)
                assert min_max_vals.shape == (self.num_parameters, 2)
                min_vals, max_vals = min_max_vals[:, 0], min_max_vals[:, 1]
                clipped_v = np.maximum(min_vals, np.minimum(max_vals, v))
                self.inverse_input_sample.assign(clipped_v)

            step += 1
            if verbose:
                # if step % 10 == 0:
                p = np.array2string(
                    prediction.numpy()[0],
                    formatter={"float_kind": lambda x: "%.2f" % x},
                )
                print(f"Step {step}, Loss: {loss.numpy()}, {p}")

        for layer in self.layers:
            layer.trainable = True

        if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):
            # invalid optimization
            return X, False

        # inverse-transform
        if transform == "square":
            zp = np.square(self.inverse_input_sample.numpy())
        elif transform == "exp":
            zp = np.exp(self.inverse_input_sample.numpy())
        elif transform == "piece_exp":
            zp = np.where(
                self.inverse_input_sample.numpy() > 0,
                self.inverse_input_sample.numpy() + 1,
                np.exp(self.inverse_input_sample.numpy()),
            )
        else:
            zp = self.inverse_input_sample.numpy()

        if np.any(np.isnan(zp)) or np.any(np.isinf(zp)) or np.any(zp == 0.0):
            # invalid optimization
            return X, False

        return zp, step

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
