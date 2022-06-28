import torch.nn as nn


class DifferentiableSGD:
    # torch.optim의 in-place 파라미터 업데이트를 우회하기위한 Differentiable SGD 옵티마이저
    # [출처](https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/differentiable_sgd.py)
    def __init__(self, model, lr=1e-3) -> None:
        self.model: nn.Module = model
        self.lr: float = lr

    def step(self) -> None:
        # 파라미터 업데이트
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

                    del model._parameters[name]
                    setattr(model, name, new_param)
                    model._parameters[name] = new_param

        update(self.model)

    def zero_grad(self, set_to_none: bool = False) -> None:
        # 파라미터의 그래디언트를 0 또는 None으로 초기화
        for param in self.model.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
