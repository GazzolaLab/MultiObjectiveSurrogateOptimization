import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)


def apply_bounds(tensor, bounds):
    return tf.stack(
        [
            tf.clip_by_value(tensor[:, i], bounds[i][0], bounds[i][1])
            for i in range(len(bounds))
        ],
        axis=1,
    )


class MLP(tf.keras.Model):
    def __init__(
        self,
        num_parameters,
        num_constraints,
        num_objectives,
        learning_rate=0.1,
        joint=True,
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.learning_rate = learning_rate
        self.joint = joint

        self.normalization_layer = tf.keras.layers.Normalization(
            input_shape=(num_parameters,)
        )
        self.hidden = tf.keras.layers.Dense(100, activation="relu")
        self.objectives_backbone = tf.keras.layers.Dense(50, activation="relu")
        self.objectives_output = tf.keras.layers.Dense(
            num_objectives, activation="linear", name="objectives"
        )
        self.constraints_output = tf.keras.layers.Dense(
            num_constraints, activation="sigmoid", name="constraints"
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=(
                {
                    "objectives": tf.keras.losses.Huber(),
                    "constraints": "binary_crossentropy",
                }
                if joint
                else "binary_crossentropy"
            ),
            metrics=(
                {
                    "objectives": ["mae"],
                    "constraints": ["acc"],
                }
                if joint
                else [self.global_accuracy]
            ),
        )

        self.inverse_input_sample = tf.Variable(
            initial_value=np.zeros([1, self.num_parameters]),
            dtype=tf.float32,
            name="inverse_input",
        )

        self.min_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="min_yR",
        )
        self.max_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="max_yR",
        )

        self._last_fit_epochs = -1

    def new(self):
        return self.__class__(
            self.num_parameters,
            self.num_constraints,
            self.num_objectives,
            self.learning_rate,
            self.joint,
        )

    def call(self, inputs):
        x = self.normalization_layer(inputs)
        x = self.hidden(x)

        ox = self.objectives_backbone(x)

        if self.joint:
            return {
                "objectives": self.objectives_output(ox),
                "constraints": self.constraints_output(x),
            }
        else:
            return self.constraints_output(x)

    def autofit(
        self,
        x,
        y,
        yC,
        epochs="auto",
        batch_size=2048,
        verbose=2,
        **kwargs,
    ):
        if epochs == "auto":
            epochs = self.autoepoch(x, y, yC, verbose=0)

        if self.joint:
            Y = {"objectives": y, "constraints": yC}
        else:
            Y = yC

        return self.fit(
            x,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    def autoeval(
        self,
        x,
        y,
        yC,
        verbose=2,
    ):
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        yC = np.nan_to_num(yC)

        if self.joint:
            Y = {"objectives": y, "constraints": yC}
        else:
            Y = yC

        return self.eval(x, Y, verbose=verbose)

    def autoepoch(self, X, y, yC, n_splits=4, timeout_samples=1e8, verbose=1):
        kf = KFold(n_splits=n_splits, shuffle=True)
        stopped_after_epochs = []
        timeout_epochs = round(timeout_samples / X.shape[0])
        epoch_increment = round(timeout_epochs / 10.0)

        def p(*args, **kwargs):
            if verbose > 0:
                print(*args, **kwargs)

        self.build(input_shape=X.shape)

        initial_weights = self.get_weights()

        p("Autoepoch cross-validation ...")
        for s, (train_index, val_index) in enumerate(kf.split(X)):
            p(f"Split {s}")
            self.set_weights(initial_weights)

            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            yC_train, yC_val = yC[train_index], yC[val_index]

            total_epochs = 0
            while total_epochs < timeout_epochs:
                p(f"{total_epochs} / {timeout_epochs}")
                if self.joint:
                    y_ = {"objectives": y_train, "constraints": yC_train}
                    val_ = (
                        X_val,
                        {
                            "objectives": y_val,
                            "constraints": yC_val,
                        },
                    )
                else:
                    y_ = yC_train
                    val_ = (X_val, yC_val)
                history = self.fit(
                    X_train,
                    y_,
                    validation_data=val_,
                    epochs=epoch_increment,
                    batch_size=2048,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor=(
                                "val_constraints_loss" if self.joint else "val_loss"
                            ),
                            patience=int(epoch_increment / 2),
                            restore_best_weights=False,
                        )
                    ],
                    verbose=verbose,
                    initial_epoch=total_epochs,
                )
                if "loss" in history.history:
                    epochs_this_round = len(history.history["loss"])
                else:
                    epochs_this_round = 1
                total_epochs += epochs_this_round

                if epochs_this_round < epoch_increment:
                    break

            p(f"Stopped after {total_epochs} for split {s}")
            stopped_after_epochs.append(total_epochs)

        m = max(stopped_after_epochs)

        self.set_weights(initial_weights)

        p(f"Average epochs: {m} for {stopped_after_epochs}")

        return int(m)

    def fit(self, x=None, y=None, *args, epochs=1, **kwargs):
        # normalize inputs
        self.normalization_layer.adapt(x)

        self._last_fit_epochs = epochs

        if self.joint:
            return super().fit(
                x,
                {
                    "objectives": self.norm_output(y["objectives"], adapt=True),
                    "constraints": y["constraints"],
                },
                *args,
                epochs=epochs,
                **kwargs,
            )
        else:
            return super().fit(x, y, *args, epochs=epochs, **kwargs)

    def norm_output(self, yR, inverse=False, adapt=False):
        if adapt:
            # mean_yR_train = np.mean(yR_train, axis=0)
            # std_yR_train = np.std(yR_train, axis=0)
            # yR_standardized = (yR_train - mean_yR_train) / std_yR_train

            self.min_yR.assign(np.min(yR, axis=0))
            self.max_yR.assign(np.max(yR, axis=0))

        if inverse:
            return yR * (self.max_yR - self.min_yR) + self.min_yR
        else:
            return (yR - self.min_yR) / (self.max_yR - self.min_yR)

    def eval(self, X_test, y_test, per_feature=False, verbose=1):
        if self.joint:
            assert (
                not per_feature
            ), "Joint model does not support per_feature evaluation"
            y_pred = self.predict(X_test, verbose=verbose)

            y_test_prime = y_test["constraints"].all(axis=1).astype(int)
            y_pred_prime = (
                (y_pred["constraints"] > 0.5).astype(int).all(axis=1).astype(int)
            )

            yR = np.nan_to_num(self.norm_output(y_pred["objectives"], inverse=True))

            return {
                "epochs": self._last_fit_epochs,
                "accuracy": accuracy_score(y_test_prime, y_pred_prime),
                "precision": precision_score(y_test_prime, y_pred_prime),
                "recall": recall_score(y_test_prime, y_pred_prime),
                "f1": f1_score(y_test_prime, y_pred_prime),
                "objective_mae": mean_absolute_error(
                    y_test["objectives"],
                    np.maximum(np.zeros_like(yR), yR),
                ),
            }
        else:
            y_prob = self.predict(X_test, verbose=verbose)
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
                "epochs": self._last_fit_epochs,
                "accuracy": accuracy_score(y_test_prime, y_pred_prime),
                "precision": precision_score(y_test_prime, y_pred_prime),
                "recall": recall_score(y_test_prime, y_pred_prime),
                "f1": f1_score(y_test_prime, y_pred_prime),
            }

    def global_accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(tf.cast(tf.reduce_all(y_true, axis=1), tf.int32), tf.float32)
        y_pred = tf.cast(tf.cast(tf.reduce_all(y_pred, axis=1), tf.int32), tf.float32)
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)

    def predict_objectives(self, X, nan_to_num=True, max_zero=True, verbose=0):
        yR = self.norm_output(
            self.predict(X, verbose=verbose)["objectives"], inverse=True
        )

        if nan_to_num:
            yR = np.nan_to_num(yR)

        if max_zero:
            yR = np.maximum(np.zeros_like(yR), yR)

        return yR

    def make_feasible(
        self,
        X,
        learning_rate=0.1,
        transform="square",
        max_iterations=100,
        max_steps_filter=None,
        use_joint_loss=False,
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
                logits = prediction
                if self.joint:
                    logits = prediction["constraints"]
                loss = loss_fn(
                    tf.constant(
                        np.ones([input_sample.shape[0], self.num_constraints]),
                        dtype=tf.float32,
                    ),
                    logits,
                )
                if self.joint and use_joint_loss:
                    # add penalty for regression targets
                    loss = loss + tf.reduce_sum(
                        tf.math.maximum(prediction["objectives"], 0)
                    )

            if iteration > max_iterations:
                break

            grads = tape.gradient(loss, input_sample)

            is_feasible = tf.math.reduce_all(logits > 0.99, axis=1)

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

        steps = steps.numpy()

        if max_steps_filter is True:
            max_steps_filter = max_iterations - 2

        if not max_steps_filter:
            return zp, steps

        # only use the samples where the steps where below cutoff
        x_filtered = np.where(
            np.tile(np.expand_dims(steps < max_steps_filter, 1), reps=X.shape[1]),
            zp,
            X,
        )

        return x_filtered, steps

    def sensitivity(self, X, reduction=lambda x: tf.reduce_mean(x, axis=0)):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            y_pred = self(X)

        if self.joint:
            return {
                k: reduction(tape.gradient(y_pred[k], X)).numpy() for k in y_pred.keys()
            }
        else:
            return reduction(tape.gradient(y_pred, X)).numpy()
