import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from scipy.stats import spearmanr
from models.utils import preprocess


huber_loss = tf.keras.losses.Huber()
log_cosh_loss = tf.keras.losses.LogCosh()


def mase_loss(y_true, y_pred):
    # https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mad = tf.reduce_mean(tf.abs(y_true - tf.reduce_mean(y_pred)))
    return mae / (mad + 1e-7)


def logscaled_huber(y_true, y_pred):
    return huber_loss(tf.math.log1p(y_true), tf.math.log1p(y_pred))


def sqrtscaled_huber(y_true, y_pred):
    return huber_loss(tf.sqrt(y_true + 1e-9), tf.sqrt(y_pred + 1e-9))


def logscaled_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred)))


def normalized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred) / tf.math.reduce_variance(y_true))


def mape(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))


def smape(y_true, y_pred):
    return 2.0 * tf.reduce_mean(
        tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred))
    )


def weighted_log_cosh_loss(y_true, y_pred):
    import tensorflow_probability as tfp

    error = y_pred - y_true
    weight = 1 / (tf.abs(y_true) * 1.0 + 1)

    weight_norm = weight / tf.reduce_sum(weight, axis=-1, keepdims=True)

    loss = weight_norm * tfp.math.log_cosh(error)

    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def apply_bounds(tensor, bounds):
    return tf.stack(
        [
            tf.clip_by_value(tensor[:, i], bounds[i][0], bounds[i][1])
            for i in range(len(bounds))
        ],
        axis=1,
    )


def acc(y_true, y_pred):
    return tf.reduce_mean(
        tf.cast(
            tf.cast(y_pred > 0.5, dtype=tf.int32)
            == tf.cast(y_true > 0.5, dtype=tf.int32),
            dtype=tf.float32,
        )
    )


def spearmanr_metric(y_true, y_pred):
    num_objectives = y_true.shape[1]
    spearman_scores = []
    for i in range(num_objectives):
        coef, _ = spearmanr(y_true[:, i], y_pred[:, i])
        spearman_scores.append(coef)
    return np.mean(spearman_scores)


class BoundsNormalization(tf.keras.layers.Layer):
    def __init__(self, xlb, xub, **kwargs):
        super().__init__(**kwargs)
        xrg = np.array(xub) - np.array(xlb)
        self.xlb = tf.convert_to_tensor(xlb, dtype=tf.float32)
        self.xrg = tf.convert_to_tensor(xrg, dtype=tf.float32)

    def call(self, inputs):
        return (inputs - self.xlb) / self.xrg

    def adapt(self, x):
        """no-op"""


class Model(tf.keras.Model):
    def __init__(
        self,
        num_parameters,
        num_constraints,
        num_objectives,
        mode="c+o",
        xlb=None,
        xub=None,
        learning_rate=0.001,
        outlier_threshold=0,
        exclude_infeasible=False,
        normalize_targets=True,
        regression_loss="mse",
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.learning_rate = learning_rate
        self.outlier_threshold = outlier_threshold
        self.exclude_infeasible = exclude_infeasible
        self.normalize_targets = normalize_targets
        if mode not in ["c+o", "c", "o"]:
            raise ValueError("Invalid mode")
        self.mode = mode
        self.xlb = xlb
        self.xub = xub

        if xlb is not None and xub is not None:
            self.input_norm_layer = BoundsNormalization(xlb, xub)
        else:
            self.input_norm_layer = tf.keras.layers.Normalization()

        self.prepare_layers()

        objective_loss = {
            "mse": "mse",
            "huber": huber_loss,
            "logcosh": log_cosh_loss,
            "weighted_logcosh": weighted_log_cosh_loss,
        }[regression_loss]

        if self.mode == "c+o":
            loss = {
                "objectives": objective_loss,
                "constraints": "binary_crossentropy",
            }
            metrics = {
                "objectives": ["mae"],
                "constraints": [acc],
            }
        elif self.mode == "c":
            loss = "binary_crossentropy"
            metrics = acc
        elif self.mode == "o":
            loss = objective_loss
            metrics = ["mae"]

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics,
        )

        self.inverse_input_sample = tf.Variable(
            initial_value=np.zeros([1, self.num_parameters]),
            dtype=tf.float32,
            name="inverse_input",
        )

        self.min_mean_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="min_mean_yR",
        )
        self.max_std_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="max_std_yR",
        )

        self._last_fit_epochs = -1

    def label(self):
        return "joint-" + self.mode

    def new(self):
        return self.__class__(
            self.num_parameters,
            self.num_constraints,
            self.num_objectives,
            self.mode,
            self.learning_rate,
            self.xlb,
            self.xub,
        )

    def build_(self, input_shape=None):
        if input_shape is None:
            input_shape = [1, self.num_parameters]
        self.call(tf.ones([1, input_shape[-1]]))

    def preprocess(self, x, y, yC=None, remove_outliers=False, nan="remove"):
        return preprocess(
            x,
            y,
            yC,
            remove_outliers=self.outlier_threshold if remove_outliers else False,
            nan=nan,
        )

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
            m = self.autoepoch(x, y, yC, verbose=0)
            epochs = np.mean(m)

        epochs = int(epochs)

        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None and self.exclude_infeasible:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]

        if self.mode == "c+o":
            Y = {"objectives": y, "constraints": yC}
        elif self.mode == "c":
            Y = yC
        elif self.mode == "o":
            Y = y

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
        x, y, yC = self.preprocess(x, y, yC)
        if self.mode == "c+o":
            Y = {"objectives": y, "constraints": yC}
        elif self.mode == "c":
            Y = yC
        elif self.mode == "o":
            Y = y

        return self.eval(x, Y, verbose=verbose)

    def autoepoch(
        self, x, y, yC, n_splits=3, timeout_samples=1e8, verbose=1, cv="time_series"
    ):
        if x.shape[0] < n_splits * 2:
            return 1

        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None and self.exclude_infeasible:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]

        kf = {"kfold": KFold, "time_series": TimeSeriesSplit}[cv](n_splits=n_splits)
        stopped_after_epochs = []
        timeout_epochs = max(25, min(round(timeout_samples / x.shape[0]), 2500))
        epoch_increment = max(10, round(timeout_epochs / 10.0))

        def p(*args, **kwargs):
            if verbose > 0:
                print(*args, **kwargs)

        self.build(input_shape=x.shape)

        initial_weights = self.get_weights()

        p("Autoepoch cross-validation ...")
        for s, (train_index, val_index) in enumerate(kf.split(x)):
            p(f"Split {s}")
            self.set_weights(initial_weights)

            X_train, X_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            yC_train, yC_val = None, None
            if yC is not None:
                yC_train, yC_val = yC[train_index], yC[val_index]

            total_epochs = 0
            while total_epochs < timeout_epochs:
                p(f"{total_epochs} / {timeout_epochs} ({epoch_increment})")
                if self.mode == "c+o":
                    y_ = {"objectives": y_train, "constraints": yC_train}
                    val_ = (
                        X_val,
                        {
                            "objectives": y_val,
                            "constraints": yC_val,
                        },
                    )
                elif self.mode == "c":
                    y_ = yC_train
                    val_ = (X_val, yC_val)
                elif self.mode == "o":
                    y_ = y_train
                    val_ = (X_val, y_val)
                history = self.fit(
                    X_train,
                    y_,
                    validation_data=val_,
                    epochs=total_epochs + epoch_increment,
                    batch_size=2048,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor=mon,
                            patience=int(epoch_increment / 2),
                            restore_best_weights=False,
                            mode="min",
                        )
                        for mon in (
                            ["val_objectives_loss", "val_constraints_loss"]
                            if self.mode == "c+o"
                            else ["val_loss"]
                        )
                    ]
                    + [tf.keras.callbacks.TerminateOnNaN()],
                    verbose=verbose,
                    initial_epoch=total_epochs,
                )
                epochs_this_round = len(history.epoch)
                total_epochs += epochs_this_round
                if epochs_this_round < epoch_increment:
                    p(
                        f"Stopping at {epochs_this_round} < {epoch_increment} (total: {total_epochs})"
                    )
                    break

            p(f"Stopped after {total_epochs} for split {s}")
            stopped_after_epochs.append(total_epochs)

        self.set_weights(initial_weights)

        return stopped_after_epochs

    def fit(self, x=None, y=None, *args, epochs=1, **kwargs):
        # normalize inputs
        self.input_norm_layer.adapt(x)

        self._last_fit_epochs = epochs

        if self.mode == "c+o":
            return super().fit(
                x,
                {
                    "objectives": self.norm_output(y["objectives"], adapt=True).numpy(),
                    "constraints": y["constraints"],
                },
                *args,
                epochs=epochs,
                **kwargs,
            )
        elif self.mode == "c":
            return super().fit(x, y, *args, epochs=epochs, **kwargs)
        else:
            return super().fit(
                x,
                self.norm_output(y, adapt=True).numpy(),
                *args,
                epochs=epochs,
                **kwargs,
            )

    def norm_output(self, yR, inverse=False, adapt=False, method="standard"):
        if not self.normalize_targets:
            return yR

        if adapt:
            if method == "minmax":
                self.min_mean_yR.assign(np.zeros([yR.shape[1]]))
                self.max_std_yR.assign(np.max(yR, axis=0))
            elif method == "standard":
                self.min_mean_yR.assign(np.mean(yR, axis=0))
                self.max_std_yR.assign(np.std(yR, axis=0))

        if method == "minmax":
            if inverse:
                return (yR + 0.5) * (
                    self.max_std_yR - self.min_mean_yR
                ) + self.min_mean_yR
            else:
                return (
                    (yR - self.min_mean_yR)
                    / (self.max_std_yR - self.min_mean_yR + tf.keras.backend.epsilon())
                ) - 0.5
        elif method == "standard":
            if inverse:
                return yR * self.max_std_yR + self.min_mean_yR
            else:
                return (yR - self.min_mean_yR) / (
                    self.max_std_yR + tf.keras.backend.epsilon()
                )
        else:
            raise ValueError("Invalid scaling method. Use 'minmax' or 'standard'.")

    def get_output_norm(self):
        if not self.normalize_targets:
            return None

        return self.min_mean_yR.numpy().tolist(), self.max_std_yR.numpy().tolist()

    def eval(self, X_test, y_test, per_feature=False, verbose=1):

        def normed(metric):
            def _w(y_true, y_pred, *args, **kwargs):
                return metric(
                    self.norm_output(y_true).numpy(),
                    self.norm_output(y_pred).numpy(),
                    *args,
                    **kwargs,
                )

            return _w

        if self.mode == "c+o":
            assert (
                not per_feature
            ), "Joint model does not support per_feature evaluation"
            y_pred = self.predict(X_test, verbose=verbose)

            y_test_prime = y_test["constraints"].all(axis=1).astype(int)
            y_pred_prime = (
                (y_pred["constraints"] > 0.5).astype(int).all(axis=1).astype(int)
            )

            return {
                "epochs": self._last_fit_epochs,
                "accuracy": float(accuracy_score(y_test_prime, y_pred_prime)),
                "precision": float(precision_score(y_test_prime, y_pred_prime)),
                "recall": float(recall_score(y_test_prime, y_pred_prime)),
                "f1": float(f1_score(y_test_prime, y_pred_prime)),
                "mdae": float(
                    normed(median_absolute_error)(
                        y_test["objectives"],
                        y_pred["objectives"],
                    )
                ),
                "mae": float(
                    normed(mean_absolute_error)(
                        y_test["objectives"],
                        y_pred["objectives"],
                    )
                ),
            }

        if self.mode == "c":
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
                "accuracy": float(accuracy_score(y_test_prime, y_pred_prime)),
                "precision": float(precision_score(y_test_prime, y_pred_prime)),
                "recall": float(recall_score(y_test_prime, y_pred_prime)),
                "f1": float(f1_score(y_test_prime, y_pred_prime)),
            }

        if self.mode == "o":
            y_pred = self.predict(X_test, verbose=verbose)
            return {
                "epochs": self._last_fit_epochs,
                "mdae": normed(median_absolute_error)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
                "r2": normed(r2_score)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
                "mae": normed(mean_absolute_error)(
                    y_test, y_pred, multioutput="raw_values"
                ).tolist(),
            }

    def global_accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(tf.cast(tf.reduce_all(y_true, axis=1), tf.int32), tf.float32)
        y_pred = tf.cast(tf.cast(tf.reduce_all(y_pred, axis=1), tf.int32), tf.float32)
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)

    def predict_objectives(self, X, nan_to_num=False, max_zero=False, verbose=0):
        yR = self.predict(X, verbose=verbose)

        if self.mode == "c+o":
            yR = yR["objectives"]

        yR = np.array(yR)

        if nan_to_num:
            yR = np.nan_to_num(yR)

        if max_zero:
            yR = np.maximum(np.zeros_like(yR), yR)

        return yR

    def predict(self, x, *args, **kwargs):
        y_pred = super().predict(x, *args, **kwargs)

        if self.mode == "c+o":
            return {
                "constraints": y_pred["constraints"],
                "objectives": self.norm_output(
                    y_pred["objectives"], inverse=True
                ).numpy(),
            }

        if self.mode == "c":
            return y_pred

        if self.mode == "o":
            return self.norm_output(y_pred, inverse=True).numpy()

    def make_feasible(
        self,
        X,
        learning_rate=0.1,
        transform=None,
        max_iterations=100,
        max_steps_filter=None,
        use_joint_loss=False,
        verbose=0,
        return_trace=False,
    ):
        if transform is None:
            if self.xlb is not None and self.xub is not None:
                transform = [(l, u) for l, u in zip(self.xlb, self.xub)]
            else:
                transform = "square"

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

        trace = [X]
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
                if self.mode == "o":
                    # simply minimize the objectives
                    loss = tf.reduce_mean(logits)
                else:
                    if self.mode == "c+o":
                        logits = prediction["constraints"]
                    loss = loss_fn(
                        tf.constant(
                            np.ones([input_sample.shape[0], self.num_constraints]),
                            dtype=tf.float32,
                        ),
                        logits,
                    )
                    if self.mode == "c+o" and use_joint_loss:
                        # add penalty for regression targets
                        loss = loss + tf.reduce_mean(prediction["objectives"])

            if iteration > max_iterations:
                break

            grads = tape.gradient(loss, input_sample)
            if grads is None:
                raise RuntimeError("Gradient computation failed")

            if self.mode != "o":
                is_feasible = tf.math.reduce_all(logits > 0.99, axis=1)

                # record number of steps for feasible samples
                steps = tf.where(is_feasible, steps, iteration)

                # zero out grads for samples that are feasible
                is_feasible_where = tf.tile(
                    tf.expand_dims(is_feasible, axis=1), [1, grads.shape[1]]
                )
                grads = tf.where(is_feasible_where, tf.zeros_like(grads), grads)

            optimizer.apply_gradients([(grads, input_sample)])

            if verbose > 0:
                print(f"Iteration {iteration}, loss = {loss.numpy()}")

            if isinstance(transform, (list, tuple)):
                input_sample.assign(apply_bounds(input_sample, transform))

            if return_trace:
                trace.append(input_sample.numpy())

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

        if return_trace:
            return trace, steps

        if max_steps_filter is True:
            max_steps_filter = max_iterations - 2

        if not max_steps_filter or self.mode == "o":
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

        if self.mode == "c+o":
            return {
                k: reduction(tape.gradient(y_pred[k], X)).numpy() for k in y_pred.keys()
            }
        else:
            return reduction(tape.gradient(y_pred, X)).numpy()


class Columnwise:
    def __init__(
        self,
        model: Model,
        num_parameters,
        num_constraints,
        num_objectives,
        mode="c+o",
        xlb=None,
        xub=None,
        **kwargs,
    ):
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.mode = mode
        self.xlb = xlb
        self.xub = xub
        self.kwargs = kwargs

        self.estimators = [
            model(
                num_parameters=self.num_parameters,
                num_constraints=self.num_constraints,
                num_objectives=1,
                mode=self.mode,
                xlb=self.xlb,
                xub=self.xub,
                **kwargs,
            )
            for _ in range(self.num_objectives)
        ]

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
        for i, estimator in enumerate(self.estimators):
            estimator.autofit(
                x,
                y[:, i : i + 1],
                yC,
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

        for i, estimator in enumerate(self.estimators):
            e = estimator.autoeval(x, y[:, i : i + 1], yC, verbose=verbose)

        # todo: merge evals
        return e

    def predict_objectives(self, X, nan_to_num=False, max_zero=False, verbose=0):
        yR = []
        for i, estimator in enumerate(self.estimators):
            yR.append(
                estimator.predict_objectives(
                    X, nan_to_num=nan_to_num, max_zero=max_zero, verbose=verbose
                )
            )

        return np.hstack(yR)

    def make_feasible(
        self,
        X,
        learning_rate=0.1,
        transform="square",
        max_iterations=100,
        max_steps_filter=None,
        use_joint_loss=False,
    ):
        pass  # TODO: merge estimate
