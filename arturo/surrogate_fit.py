from collections import defaultdict
import torch



class SurrogateFit:
    def __init__(self, config):
        self.mf_state = defaultdict(lambda: {})
        self.config = config

    def step(self, p):
        # optimizer state is not used here, only need the parameter p and its grad
        mf_state = self.mf_state[p]
        if "quad_term" not in mf_state:
            self._initialize_surrogate_fit(p)

        # 00: top left, 01: top right/bottom left , 02: bottom right of the filter cov matrix
        kv_00 = mf_state["kv_00"]
        kv_01 = mf_state["kv_01"]
        kv_11 = mf_state["kv_11"]
        quad_term = mf_state["quad_term"]
        lin_term = mf_state["lin_term"]

        mf_state["lin_term"], mf_state["quad_term"], mf_state["kv_00"], mf_state["kv_01"], mf_state["kv_11"] = _filter_update(
            kv_00, kv_01, kv_11, quad_term, lin_term, p.data, p.grad, self.config["dynamic_model_noise_lin"],
            self.config["dynamic_model_noise_quad"], self.config["dynamic_model_noise_top_right"], self.config["measurement_noise"]
        )


    def _initialize_surrogate_fit(self, p):
        """
        Sets the state for the parameter p.
        :param p: parameter
        :param group: group which p belongs to. It stores the hyperparameters
        :return: None
        """
        state = self.mf_state[p]
        state["quad_term"] = torch.ones_like(p.data) * 0.1
        state["lin_term"] = p.grad.clone()
        # start filter covariance
        state["kv_00"] = torch.ones_like(p.data).mul_(self.config["start_filter_cov"])
        state["kv_11"] = state["kv_00"].clone()
        state["kv_01"] = torch.zeros_like(p.data)
        state["gain_0"] = torch.zeros_like(p.data)
        state["gain_1"] = torch.zeros_like(p.data)
        state["s"] = torch.zeros_like(p.data)
        state["temp"] = torch.zeros_like(p.data)

        d_noise = self.config["dynamic_model_noise"]
        self.config["dynamic_model_noise_lin"] = d_noise
        self.config["dynamic_model_noise_quad"] = d_noise
        self.config["dynamic_model_noise_top_right"] = 0.0

    def get_gradient(self, p):
        mf_state = self.mf_state[p]
        return p.data * mf_state["quad_term"] + mf_state["lin_term"]

    def get_hessian(self, p):
        return self.mf_state[p]["quad_term"]

    def get_lin_term(self, p):
        mf_state = self.mf_state[p]
        return mf_state["lin_term"]

    def get_quad_term(self, p):
        mf_state = self.mf_state[p]
        return mf_state["quad_term"]

    def get_gains(self, p):
        mf_state = self.mf_state[p]
        return mf_state["gain_0"], mf_state["gain_1"]


@torch.jit.script
def _filter_update(kv_00, kv_01, kv_11, quad_term, lin_term, p_data, p_grad, dm_lin, dm_quad, dm_top_right, mn):
    # 1) P_k += Q
    kv_00 += dm_lin
    kv_11 += dm_quad
    kv_01 += dm_top_right
    # 2a) gain = P_k * H_k
    gain_0 = kv_00 + kv_01 * p_data
    gain_1 = kv_01 + kv_11 * p_data
    # 2b) S = H P H + sigma^2
    s = gain_0 + gain_1 * p_data + mn
    inv_s = 1 / s
    # 3) K = P * H * S^-1
    gain_0 *= inv_s
    gain_1 *= inv_s
    # 4) m += K (y - H m)
    delta = p_grad - p_data * quad_term - lin_term
    lin_term += gain_0 * delta
    quad_term += gain_1 * delta
    # 5) P -= K S K^T
    kv_00 -= gain_0 * (s * gain_0)
    kv_01 -= gain_0 * (s * gain_1)
    kv_11 -= gain_1 * (s * gain_1)
    return lin_term, quad_term, kv_00, kv_01, kv_11
