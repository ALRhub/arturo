import time

import torch
from torch.optim.optimizer import Optimizer

from .surrogate_fit import SurrogateFit
from .dual_solver import DualSolver


class Arturo(Optimizer):
    def __init__(self, params, config):
        super().__init__(params, config)
        # self.kl_bound = config["kl_bound"]
        self.reg_weight = config["reg_weight"]
        self.eps = config.get("eps", 1.0e-8)
        self.clip_var = config.get("clip_var", False)
        if not self.clip_var:
            self.eps = None
        self.weight_decay = config.get("weight_decay", 0.0)
        self.config = config
        self.etas = []
        self.iteration_counter = 1

        self.surrogate_fit = SurrogateFit(config)

        # initialize
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["mean"] = torch.clone(p.data)
                # use precision instead of variance since precision is used in the formulas
                state["prec"] = torch.ones_like(p) * 1 / config.get("var_init", 0.01)

        self.solver = DualSolver(eta_tol=self.config.get("eta_tol", 0.5),
                                 param_groups=self.param_groups,
                                 reg_weight=self.reg_weight,
                                 prior_prec=self.config["prior_prec"],
                                 kl_bound_getter=lambda: self.param_groups[0]["lr"],
                                 mu_getter=lambda p: self.state[p]["mean"],
                                 prec_getter=lambda p: self.state[p]["prec"],
                                 quad_term_getter=lambda p: self.surrogate_fit.get_quad_term(p),
                                 lin_term_getter=lambda p: self.surrogate_fit.get_lin_term(p),
                                 clip_eps=self.eps,
                                 nu=self.config.get("nu", 0.0))

    def step(self, closure=None):
        if closure is not None:
            print("Warning: GradientMORE Optimizer: closure is not implemented.")

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                # weight decay
                if self.weight_decay != 0:
                    # group["lr"] is the kl bound, so the scheduling also affects the weight decay as it is usual for weight decay algos
                    p.data.sub_(p.data, alpha=group["lr"] * self.weight_decay)
                # update surrogate
                self.surrogate_fit.step(p)
        # solve dual
        current_eta = self.solver.solve_dual()
        self.etas.append(current_eta)
        # update params
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                state = self.state[p]
                dual_solver_state = self.solver.dual_solver_state[p]
                self.solver.primal_solution(p, current_eta)
                state["mean"] = dual_solver_state["new_mu"]
                state["prec"] = dual_solver_state["new_prec"]
                # insert into model
                p.data = dual_solver_state["new_mu"]

    def aggregate_etas_and_reset(self):
        mean_eta = torch.mean(torch.tensor(self.etas))
        self.etas = []
        return mean_eta
