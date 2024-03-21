from interface.dmosopt import Dmosopt


class Sopt(Dmosopt):

    def version_joint_model(
        self,
        scope=None,
        joint=True,
        feasibility_solving=False,
        feasibility_max_iterations=50,
        feasibility_use_joint_loss=True,
        feasibility_max_steps_filter=True,
    ):
        if scope is None:
            scope = [
                "objective",
                "feasiblity",
            ]
        return {
            "dopt_params": {
                "surrogate_custom_training": "models.ops.mlp",
                "surrogate_custom_training_kwargs": {
                    "scope": scope,
                    "joint": joint,
                    "feasibility_solving": feasibility_solving,
                    "feasibility_max_iterations": feasibility_max_iterations,
                    "feasibility_use_joint_loss": feasibility_use_joint_loss,
                    "feasibility_max_steps_filter": feasibility_max_steps_filter,
                },
            }
        }

    def version_dynamic_sampling(
        self,
        max_iterations=10,
        stop_condition="f1>0.4",
        feasibility_solving=False,
        feasibility_max_iterations=50,
        feasibility_use_joint_loss=True,
        feasibility_max_steps_filter=True,
    ):
        return {
            "dopt_params": {
                "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                "dynamic_initial_sampling_kwargs": {
                    "max_iterations": max_iterations,
                    "stop_condition": stop_condition,
                    "feasibility_solving": feasibility_solving,
                    "feasibility_max_iterations": feasibility_max_iterations,
                    "feasibility_use_joint_loss": feasibility_use_joint_loss,
                    "feasibility_max_steps_filter": feasibility_max_steps_filter,
                },
            }
        }
