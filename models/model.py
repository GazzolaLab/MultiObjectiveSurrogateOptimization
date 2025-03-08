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
from collections import deque


class MovingAverageEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, window_size=5, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size)

    def get_monitor_value(self, logs):
        current_val = logs.get(self.monitor)
        if current_val is None:
            return None

        self.loss_window.append(current_val)

        if len(self.loss_window) == self.window_size:
            return np.mean(self.loss_window)
        return current_val


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


def distance_weighted_mse(y_true, y_pred):
    weight = 1 / (1 + tf.abs(y_true))
    weighted_squared_error = weight * tf.square(y_true - y_pred)
    loss = tf.reduce_sum(weighted_squared_error, axis=-1)
    return tf.reduce_mean(loss)


def relative_error_loss(y_true, y_pred, epsilon=1e-7):
    y_true_safe = tf.where(tf.abs(y_true) > epsilon, y_true, epsilon * tf.sign(y_true))
    relative_errors = tf.abs((y_true - y_pred) / y_true_safe)
    return tf.reduce_mean(tf.reduce_mean(relative_errors, axis=-1))


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
            tf.cast(tf.cast(y_pred, tf.float32) > 0.5, dtype=tf.int32)
            == tf.cast(tf.cast(y_true, tf.float32) > 0.5, dtype=tf.int32),
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


def distance_maximization_loss(input_sample, X_, weight=1.0, epsilon=1e-12):
    # input_sample shape: [batch_size, num_parameters]
    # X_ shape: [n_samples, num_parameters]

    # broad cast line up
    input_expanded = tf.expand_dims(
        input_sample, axis=1
    )  # [batch_size, 1, num_parameters]
    X_expanded = tf.expand_dims(X_, axis=0)  # [1, n_samples, num_parameters]

    # pairwise distances => [batch_size, n_samples]
    distances = tf.sqrt(
        tf.reduce_sum(tf.square(input_expanded - X_expanded), axis=-1) + epsilon
    )

    batch_distances = tf.reduce_mean(distances, axis=1)  # [batch_size]

    loss = -tf.reduce_mean(batch_distances)  # []

    return weight * loss


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
        normalize_targets="range",
        gradnorm=False,
        regression_loss="mse",
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        self.num_parameters = num_parameters
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.learning_rate = learning_rate
        self.outlier_threshold = outlier_threshold
        self.regression_loss = regression_loss
        self.exclude_infeasible = exclude_infeasible
        self.normalize_targets = normalize_targets
        if mode not in ["c+o", "c", "o"]:
            raise ValueError("Invalid mode")
        self.use_gradnorm = gradnorm and (mode != "c")
        self.mode = mode
        self.xlb = xlb
        self.xub = xub
        self.X_ = None
        self.X_raw_ = None
        self.y_ = None
        self.y_raw_ = None
        self.y_norm_ = None
        self.yC_ = None
        self.yC_raw_ = None

        if xlb is not None and xub is not None:
            self.input_norm_layer = BoundsNormalization(xlb, xub)
        else:
            self.input_norm_layer = tf.keras.layers.Normalization()

        self.gradnorm_layers = []

        self.timestep = None
        if self.use_gradnorm:
            self.objective_weights = tf.Variable(
                [1] * self.num_objectives,
                name="objective_weights",
                dtype=tf.float32,
                trainable=False,
            )
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.loss_t_zero = tf.Variable(
                [0] * self.num_objectives, trainable=False, dtype=tf.float32
            )
            self.timestep = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.prepare_layers()

        self.autocompile()

        self.inverse_input_sample = tf.Variable(
            initial_value=np.zeros([1, self.num_parameters]),
            dtype=tf.float32,
            name="inverse_input",
            trainable=False,
        )

        self.min_mean_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="min_mean_yR",
            trainable=False,
        )
        self.max_std_yR = tf.Variable(
            initial_value=np.zeros([num_objectives]),
            dtype=tf.float32,
            name="max_std_yR",
            trainable=False,
        )

        self._last_fit_epochs = -1

    def objective_loss(self, y_true, y_pred, alpha=1, beta=0.01, verbose=False):
        mins = tf.reduce_min(self.y_norm_, axis=0)
        maxs = tf.reduce_max(self.y_norm_, axis=0)

        weights = 1  # / (maxs + 1e-7)

        if verbose:
            print("mins", mins)
            print("maxs", maxs)

            print(y_true, "y_true")
            print(y_pred, "y_pred")

        # move to floor
        ytrue = y_true - mins
        ypred = y_pred - mins

        if verbose:
            print(ytrue, "ytrue")
            print(ypred, "ypred")

        # high weights for low values
        rerr = tf.abs(ytrue - ypred) / (alpha * tf.abs(ytrue) + beta)

        if verbose:
            print(rerr, "rerr")

        rerr_ = tf.reduce_mean(rerr, axis=0)

        return tf.reduce_mean(weights * rerr_)

    def train_step(self, data):
        if not self.use_gradnorm:
            return super().train_step(data)

        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            W = sum([l.trainable_variables for l in self.gradnorm_layers], [])

            losses = [
                self.compute_loss(y=y[i], y_pred=y_pred[i])
                for i in range(self.num_objectives)
            ]
            # save t=0 losses
            self.loss_t_zero.assign_add(tf.where(self.timestep == 0, losses, 0))
            weighted_losses = tf.multiply(losses, self.objective_weights)

            # compute weighted loss norms
            grads = [
                tape.gradient(weighted_losses[i], W) for i in range(self.num_objectives)
            ]
            G_W = [
                tf.norm(
                    tf.concat(
                        [tf.reshape(grad, [-1]) for grad in gs if grad is not None],
                        axis=0,
                    ),
                    ord=2,
                )
                for i, gs in enumerate(grads)
            ]

            # compute training rates
            L_i_tilde = [
                losses[i] / self.loss_t_zero[i] for i in range(self.num_objectives)
            ]
            L_i_tilde_mean = tf.reduce_mean(L_i_tilde, name="L_i_tilde_mean")
            r_i_t = [l / (L_i_tilde_mean + 1e-9) for l in L_i_tilde]

            # compute gradnorm loss
            alpha = 0.5
            L_grad = 0
            for i in range(self.num_objectives):
                L_grad += tf.abs(
                    G_W[i]
                    - tf.stop_gradient(
                        tf.reduce_mean(G_W, name="G_W_mean") * tf.pow(r_i_t[i], alpha)
                    )
                )

            # loss weight update
            grad_norm_grads = tf.gradients(L_grad, self.objective_weights)[0]
            self.objective_weights.assign_sub(
                tf.where(self.timestep > 0, 1e-4 * grad_norm_grads, 0)
            )
            # normalize
            self.objective_weights.assign(
                self.objective_weights
                / (tf.reduce_sum(self.objective_weights) + 1e-9)
                * self.num_objectives
            )

            # module gradient computation
            total_loss = tf.reduce_mean(weighted_losses)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

        del tape

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.timestep.assign_add(1)

        self.loss_tracker.update_state(total_loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(self.loss_tracker.result())
            else:
                metric.update_state(y, y_pred)

        return {
            **{m.name: m.result() for m in self.metrics},
            # **{
            #     "g0": grad_norm_grads[0],
            #     "g1": grad_norm_grads[1],
            #     "g2": grad_norm_grads[2],
            #     "g3": grad_norm_grads[3],
            #     "r0": r_i_t[0],
            #     "r1": r_i_t[1],
            #     "r2": r_i_t[2],
            #     "r3": r_i_t[3],
            #     "w0": self.objective_weights[0],
            #     "w1": self.objective_weights[1],
            #     "w2": self.objective_weights[2],
            #     "w3": self.objective_weights[3],
            #     "w_sum": tf.reduce_sum(self.objective_weights)
            # },
        }

    def test_step(self, data):
        if not self.use_gradnorm:
            return super().test_step(data)
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

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

    def build(self, input_shape=None):
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

    def autocompile(self):
        try:
            objective_loss = {
                "mae": "mae",
                "mse": "mse",
                "huber": huber_loss,
                "logcosh": log_cosh_loss,
                "weighted_logcosh": weighted_log_cosh_loss,
                "distance_weighted_mse": distance_weighted_mse,
                "relative_error": relative_error_loss,
            }[self.regression_loss]
        except KeyError:
            objective_loss = self.objective_loss

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics,
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
            m = self.autoepoch(x, y, yC, verbose=1)
            print("Automatic epochs: ", m, " -> ", np.mean(m))
            epochs = np.mean(m)
        else:
            self.build(input_shape=x.shape)

        epochs = int(epochs)

        self.X_raw_ = x
        self.y_raw_ = y
        self.yC_raw_ = yC

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
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                # tf.keras.callbacks.ReduceLROnPlateau(monitor="loss"),
            ],
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
        self, x, y, yC, n_splits=3, timeout_samples=1e8, verbose=1, cv="kfold"
    ):
        if x.shape[0] < n_splits * 2:
            return [1]

        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None and self.exclude_infeasible:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]

        if x.shape[0] < n_splits * 2:
            return [1]

        kf = {"kfold": KFold, "time_series": TimeSeriesSplit}[cv](n_splits=n_splits)
        stopped_after_epochs = []
        timeout_epochs = max(25, min(round(timeout_samples / x.shape[0]), 10000))
        epoch_increment = max(10, round(timeout_epochs / 10.0))

        def p(*args, **kwargs):
            if verbose > 0:
                print(*args, **kwargs)

        self.build(input_shape=x.shape)

        initial_weights = self.get_weights()

        p("Autoepoch cross-validation ...")
        for s, (train_index, val_index) in enumerate(kf.split(x)):
            p(f"Split {s}")
            self.autocompile()
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
                            patience=250,
                            restore_best_weights=False,
                            mode="min",
                        )
                        for mon in (
                            [
                                "val_objectives_loss",  # "val_constraints_loss"
                            ]
                            if self.mode == "c+o"
                            else ["val_mae"]  # "val_loss"
                        )
                    ]
                    + [
                        tf.keras.callbacks.TerminateOnNaN(),
                        # tf.keras.callbacks.ReduceLROnPlateau(
                        #     factor=0.5,
                        #     monitor=(
                        #         "val_objectives_loss"
                        #         if self.mode == "c+o"
                        #         else "val_loss"
                        #     )
                        # ),
                    ],
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

        self.autocompile()
        self.set_weights(initial_weights)

        return stopped_after_epochs

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        *args,
        **kwargs,
    ):
        self.X_ = x
        if self.mode == "c+o":
            self.y_ = y["objectives"]
            self.yC_ = y["constraints"]
        elif self.mode == "c":
            self.y_ = None
            self.yC_ = y
        elif self.mode == "o":
            self.y_ = y
            self.yC_ = None

        if self.timestep is not None:
            self.timestep.assign(0)

        # normalize inputs
        self.input_norm_layer.adapt(x)

        self._last_fit_epochs = epochs

        if self.mode == "c+o":
            self.y_norm_ = self.norm_output(y["objectives"], adapt=True).numpy()
            if validation_data is not None:
                validation_data = (
                    validation_data[0],
                    {
                        "objectives": self.norm_output(
                            validation_data[1]["objectives"]
                        ).numpy(),
                        "constraints": validation_data[1]["constraints"],
                    },
                )
            return super().fit(
                x,
                {
                    "objectives": self.y_norm_,
                    "constraints": y["constraints"],
                },
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )
        elif self.mode == "c":
            return super().fit(
                x,
                y,
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )
        else:
            self.y_norm_ = self.norm_output(y, adapt=True).numpy()
            if validation_data is not None:
                validation_data = (
                    validation_data[0],
                    self.norm_output(validation_data[1]).numpy(),
                )
            return super().fit(
                x,
                self.y_norm_,
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data,
                *args,
                **kwargs,
            )

    def norm_output(self, yR, inverse=False, adapt=False, method=None):
        if method is None:
            method = self.normalize_targets

        if method is False:
            return tf.constant(yR)

        if adapt:
            if "max" in method or method == "range":
                if "0" in method:
                    self.min_mean_yR.assign(np.zeros([yR.shape[1]]))
                else:
                    self.min_mean_yR.assign(np.min(yR, axis=0))
                self.max_std_yR.assign(np.max(yR, axis=0))
            elif method == "standard":
                self.min_mean_yR.assign(np.mean(yR, axis=0))
                self.max_std_yR.assign(np.std(yR, axis=0))
            elif method == "log":
                pass

        if "max" in method:
            centering = 0.0
            if "c" in method:
                centering = 0.5
            if inverse:
                return (yR + centering) * (
                    self.max_std_yR - self.min_mean_yR
                ) + self.min_mean_yR
            else:
                return (
                    (yR - self.min_mean_yR)
                    / (self.max_std_yR - self.min_mean_yR + tf.keras.backend.epsilon())
                ) - centering
        elif method == "range":
            top = tf.reduce_max(self.max_std_yR)
            bottom = tf.reduce_max(self.min_mean_yR)
            if inverse:
                exps = tf.exp(yR) - 1
                scaled = (exps - bottom) / (top - bottom + tf.keras.backend.epsilon())
                return scaled * (self.max_std_yR - self.min_mean_yR) + self.min_mean_yR
            else:
                normalized = (yR - self.min_mean_yR) / (
                    self.max_std_yR - self.min_mean_yR + tf.keras.backend.epsilon()
                )
                upscaled = normalized * (top - bottom) + bottom
                return tf.math.log1p(upscaled)
        elif method == "standard":
            if inverse:
                return yR * self.max_std_yR + self.min_mean_yR
            else:
                return (yR - self.min_mean_yR) / (
                    self.max_std_yR + tf.keras.backend.epsilon()
                )
        elif "log" in method:
            if inverse:
                if method == "log_":
                    return tf.constant(yR)
                return tf.exp(yR) - 1
            else:
                return tf.math.log1p(yR)
        else:
            raise ValueError(f"Invalid scaling method: {method}.")

    def get_output_norm(self):
        if self.normalize_targets is False:
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

    def raw_predict(self, x, *args, **kwargs):
        return super().predict(x, *args, **kwargs)

    def make_feasible(
        self,
        X,
        learning_rate=0.001,
        transform=None,
        max_iterations=1000,
        detect_plateau=True,
        max_steps_filter=None,
        targets="objective",
        zero_infeasible=False,
        verbose=1,
        return_trace=False,
        renorm=False,
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

        trace = [X]
        loss_history = []
        iteration = 0
        while True:
            losses = []
            with tf.GradientTape(persistent=True) as tape:
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

                logits_o = None
                logits_c = None
                if self.mode == "c+o":
                    logits_o = prediction["objectives"]
                    logits_c = prediction["constraints"]
                elif self.mode == "o":
                    logits_o = prediction
                elif self.mode == "c":
                    logits_c = prediction

                if "o" in self.mode and "objective" in targets:
                    y_ = tf.constant(self.y_)
                    o = self.norm_output(logits_o, inverse=True)

                    nadir = tf.reduce_max(tf.concat([o, y_], axis=0), axis=0)

                    front = o / (nadir + 1e-8)
                    reference = tf.ones_like(nadir) * 1.1

                    dominated_space = tf.maximum(reference - front, 0)
                    volume = tf.reduce_prod(dominated_space, axis=-1)
                    score = tf.reduce_sum(-1 * volume)

                    prefix = -1 if "-objective" in targets else 1
                    losses.append(prefix * score)

                if "c" in self.mode and "constraint" in targets:
                    losses.append(
                        tf.keras.losses.BinaryFocalCrossentropy()(
                            tf.constant(
                                prefix
                                * np.ones(
                                    [input_sample.shape[0], self.num_constraints]
                                ),
                                dtype=tf.float32,
                            ),
                            logits_c,
                        )
                    )

                if self.X_raw_ is not None and "distance" in targets:
                    # normalize input_sample and X_ to [0,1] using bounds
                    xlb = tf.constant(self.xlb)
                    xub = tf.constant(self.xub)
                    r = xub - xlb
                    # maximize pairwise distance to ensure exploration
                    losses.append(
                        distance_maximization_loss(
                            (input_sample - xlb) / r, (self.X_raw_ - xlb) / r
                        )
                    )

            derivatives = [tape.gradient(lt, input_sample) for lt in losses]
            del tape

            if renorm and isinstance(transform, (list, tuple)):
                rg = tf.constant([u - l for l, u in transform])
                derivatives = [dl / rg for dl in derivatives]

            # balance via gradient norms
            if len(losses) > 1:
                norms = [tf.norm(g) for g in derivatives]
                ref_norm = norms[0]
                factors = [ref_norm / (norm + 1e-8) for norm in norms]
                loss = tf.reduce_sum([w * l for w, l in zip(factors, losses)])
                grads = tf.reduce_sum([w * l for w, l in zip(factors, derivatives)])
            else:
                loss = losses[0]
                grads = derivatives[0]

            if iteration > max_iterations:
                break

            loss_history.append(loss.numpy())

            plateau_window = 50
            if detect_plateau and len(loss_history) > plateau_window:
                recent_losses = loss_history[-plateau_window:]
                # iqr
                q1 = np.percentile(recent_losses, 25)
                q3 = np.percentile(recent_losses, 75)
                iqr = q3 - q1

                # break if IQR is very small relative to the median
                median = np.median(recent_losses)
                relative_iqr = iqr / abs(median) if median != 0 else iqr

                if relative_iqr < 0.01:
                    if verbose > 0:
                        print(
                            f"Loss is plateauing, stopping early after {len(loss_history)} steps."
                        )
                    break

            if zero_infeasible and self.mode != "o":
                is_feasible = tf.math.reduce_all(logits_c > 0.99, axis=1)

                # record number of steps for feasible samples
                steps = tf.where(is_feasible, steps, iteration)

                # zero out grads for samples that are feasible
                is_feasible_where = tf.tile(
                    tf.expand_dims(is_feasible, axis=1), [1, grads.shape[1]]
                )
                grads = tf.where(is_feasible_where, tf.zeros_like(grads), grads)

            optimizer.apply_gradients([(grads, input_sample)])

            if verbose > 0:
                try:
                    preds = np.mean(prediction, axis=0)
                    objs = np.mean(self.norm_output(prediction, inverse=True), axis=0)
                    print(
                        f"Iteration {iteration}, loss = {loss.numpy()}, logits={preds}, objectives={objs}"
                    )
                except:
                    pass

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
            return trace, loss_history

        if max_steps_filter is True:
            max_steps_filter = max_iterations - 2

        if not max_steps_filter or self.mode == "o":
            return zp, loss_history

        # only use the samples where the steps where below cutoff
        x_filtered = np.where(
            np.tile(np.expand_dims(steps < max_steps_filter, 1), reps=X.shape[1]),
            zp,
            X,
        )

        return x_filtered, loss_history

    def sensitivity(
        self, X, reduction=lambda x: tf.reduce_mean(tf.math.abs(x), axis=0), renorm=False
    ):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            y_pred = self(X)

        if self.mode == "o":
            y_pred = {
                "objectives": y_pred,
                "constraints": None,
            }
        elif self.mode == "c":
            y_pred = {
                "objectives": None,
                "constraints": y_pred,
            }

        sens = {}
        for k in y_pred.keys():
            if y_pred[k] is None:
                sens[k] = None
                continue
            g = tape.gradient(y_pred[k], X)

            # adjust by chain-rule
            if renorm:
                g = g / self.input_norm_layer.xrg

            sens[k] = reduction(g).numpy()

        return sens


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
