from collections import defaultdict
import torch


class DualSolver:
    def __init__(self, eta_tol, param_groups, reg_weight, prior_prec, kl_bound_getter, mu_getter, prec_getter, quad_term_getter,
                 lin_term_getter,
                 clip_eps, nu):
        self.param_groups = param_groups
        self.eta_tol = eta_tol
        self.mu_getter = mu_getter
        self.prec_getter = prec_getter
        self.quad_term_getter = quad_term_getter
        self.lin_term_getter = lin_term_getter
        self.reg_weight = reg_weight
        self.kl_bound_getter = kl_bound_getter
        self.clip_eps = clip_eps

        for group in self.param_groups:
            for p in group["params"]:
                if mu_getter(p).is_cuda:
                    self.device = "cuda"
                else:
                    self.device = "cpu"

        self._old_eta = torch.tensor(-1.0).to(self.device)

        self.dual_solver_state = defaultdict(lambda: {})

        for group in self.param_groups:
            for p in group['params']:
                state = self.dual_solver_state[p]
                state["new_mu"] = torch.zeros_like(p.data)
                state["new_prec"] = torch.zeros_like(p.data)
                state["temp"] = torch.zeros_like(p.data)

        self.prior_prec = prior_prec
        self.reg_weight_times_prior_prec = self.reg_weight * self.prior_prec
        self.nu = nu

    def primal_solution(self, p, trial_eta):
        state = self.dual_solver_state[p]
        state["new_mu"], state["new_prec"] = _primal_solution(trial_eta, self.mu_getter(p), self.prec_getter(p), self.quad_term_getter(p),
                                                              self.lin_term_getter(p), self.reg_weight, self.reg_weight_times_prior_prec,
                                                              self.nu)


    def d_dual_d_eta(self, trial_eta):
        result = torch.tensor([0.0], device=self.device)
        for group in self.param_groups:
            for p in group['params']:
                state = self.dual_solver_state[p]
                self.primal_solution(p, trial_eta)
                # mahanalobis dist between new and old dist
                result.add_(_mahanalobis_bound(state["new_mu"], self.mu_getter(p), self.prec_getter(p)))
        result.mul_(0.5)
        result.sub_(self.kl_bound_getter())
        return result

    def solve_dual(self):
        total_max_eta = 10000
        if self._old_eta >= 0.0:
            lower_eta = self._old_eta / 3
            upper_eta = torch.clip(3 * self._old_eta, max=total_max_eta)
        else:
            lower_eta = torch.tensor([0.0]).to(self.device)
            upper_eta = torch.tensor([total_max_eta]).to(self.device)
        self.eta_iterations = 0
        while upper_eta - lower_eta > self.eta_tol:
            self.eta_iterations += 1
            mid_point = (upper_eta + lower_eta) / 2
            d_dual = self.d_dual_d_eta(mid_point)
            if d_dual > 0.0:
                # midpoint too small -> set as lower point
                lower_eta = mid_point
            else:
                # midpoint too big --> set as upper point
                upper_eta = mid_point

            if abs(d_dual) < 0.1 * self.kl_bound_getter():
                break

        self._old_eta = (upper_eta + lower_eta) / 2
        return self._old_eta


@torch.jit.script
def _mahanalobis_bound(new_mu, old_mu, prec):
    return torch.sum((old_mu - new_mu) ** 2 * prec)

@torch.jit.script
def _primal_solution(trial_eta, mu, prec, quad_term, lin_term, reg_weight, reg_weight_times_prior_prec, nu):
    temp = prec * trial_eta
    temp2 = quad_term + reg_weight_times_prior_prec
    new_mu = (temp * mu - lin_term) / (temp2 + temp)
    new_prec = (temp2 + prec * nu) / (reg_weight + nu)
    return new_mu, new_prec


