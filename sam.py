import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, use_grad_norm, zero_grad=False):
        grad_norm = self._grad_norm()
        ew_norm_squared_total = 0.0  # Initialize the total squared norm of e_w
        for group in self.param_groups:
            if use_grad_norm:
                scale = group["rho"] / (grad_norm + 1e-12)
            else:
                scale = torch.tensor(group["rho"]).to(self.param_groups[0]["params"][0].device)

            for p in group["params"]: #note that 'p' (normally) is parameters vector for a certain layer, not in dividual parameter
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.sub_(e_w)  # descend to the local minimum "w - e(w)" #AACE
                
                ew_norm_squared_total += e_w.norm(p=2).pow(2)  # Accumulate the squared norm of e_w
        
        ew_norm = (ew_norm_squared_total ** 0.5).item()  # Calculate the total norm of e_w
        
        if zero_grad: self.zero_grad()
        return grad_norm.item(), ew_norm

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups