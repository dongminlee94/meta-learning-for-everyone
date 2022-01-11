"""
Differentiable Optimizer for higher-ogder optimization
"""

import torch.nn as nn


class DifferentiableSGD:
    """Differenctialble SGD avoiding parameter's in-place update of torch.optim"""

    def __init__(self, model, lr=1e-3) -> None:
        self.model: nn.Module = model
        self.lr: float = lr

    def step(self) -> None:
        """Update papareter"""
        module_set = set()

        def update(model: nn.Module) -> None:
            for sub_module in model.children():
                if sub_module not in module_set:
                    module_set.add(sub_module)
                    update(sub_module)

            params = list(model.named_parameters())
            for name, param in params:
                if "." not in name:
                    if param.grad is None:
                        continue

                    new_param = param.add(param.grad, alpha=-self.lr)

                    del model._parameters[name]  # pylint: disable=protected-access
                    setattr(model, name, new_param)
                    model._parameters[name] = new_param  # pylint: disable=protected-access

        update(self.model)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Set grdients of parameters to zero or None"""

        for param in self.model.parameters():
            if param.grad is not None:

                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
