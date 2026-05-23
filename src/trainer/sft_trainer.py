from transformers import Trainer


class SmolVLMSFTTrainer(Trainer):

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr_by_module = {
            module_name: learning_rate
            for module_name, learning_rate in (
                ("vision_model", self.args.vision_lr),
                ("connector", self.args.connector_lr),
            )
            if learning_rate is not None
        }
        if not lr_by_module:
            return super().create_optimizer()

        trainable_named_parameters = [
            (name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad
        ]

        def uses_weight_decay(parameter_name: str) -> bool:
            normalized_name = parameter_name.lower()
            return not (
                parameter_name.endswith(".bias")
                or "norm" in normalized_name
                or "layernorm" in normalized_name
                or "layer_norm" in normalized_name
            )

        def belongs_to_module(parameter_name: str, module_name: str) -> bool:
            return module_name in parameter_name

        def append_group(parameters, *, weight_decay: float, learning_rate: float | None = None):
            if not parameters:
                return
            group = {
                "params": parameters,
                "weight_decay": weight_decay,
            }
            if learning_rate is not None:
                group["lr"] = learning_rate
            optimizer_grouped_parameters.append(group)

        optimizer_grouped_parameters = []
        special_module_names = tuple(lr_by_module)
        base_parameters = [
            (name, parameter)
            for name, parameter in trainable_named_parameters
            if not any(belongs_to_module(name, module_name) for module_name in special_module_names)
        ]
        append_group(
            [parameter for name, parameter in base_parameters if uses_weight_decay(name)],
            weight_decay=self.args.weight_decay,
        )
        append_group(
            [parameter for name, parameter in base_parameters if not uses_weight_decay(name)],
            weight_decay=0.0,
        )

        for module_name, learning_rate in lr_by_module.items():
            module_parameters = [
                (name, parameter)
                for name, parameter in trainable_named_parameters
                if belongs_to_module(name, module_name)
            ]
            append_group(
                [parameter for name, parameter in module_parameters if uses_weight_decay(name)],
                weight_decay=self.args.weight_decay,
                learning_rate=learning_rate,
            )
            append_group(
                [parameter for name, parameter in module_parameters if not uses_weight_decay(name)],
                weight_decay=0.0,
                learning_rate=learning_rate,
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args,
            self.model,
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
