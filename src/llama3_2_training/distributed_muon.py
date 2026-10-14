import torch
import torch.distributed as dist


class DistributedMuon:
    """把 Muon 的 Newton–Schulz 計算分散到各 rank。

    DDP 之後每個 rank 都有完整的平均梯度；原本每個 rank 都對所有矩陣跑一次 NS（重複 world 次）。
    這裡每個矩陣只由一個 owner rank 更新，更新完再 broadcast 給其他 rank，結果與單機 Muon 相同。
    optimizer state（momentum）只存在 owner 上，所以 checkpoint 要每個 rank 各存一份（sharded）。
    """

    sharded = True

    def __init__(self, params, rank, world, **muon_kwargs):
        self.params = list(params)
        self.rank, self.world = rank, world
        # 依大小貪婪分配，讓每個 rank 的 NS 工作量接近；順序固定，所有 rank 算出同一份分配
        loads = [0] * world
        self.owner = [0] * len(self.params)
        for i in sorted(range(len(self.params)), key=lambda i: -self.params[i].numel()):
            r = loads.index(min(loads))
            self.owner[i] = r
            loads[r] += self.params[i].numel()
        own = [p for p, o in zip(self.params, self.owner) if o == rank]
        self.opt = torch.optim.Muon(own, **muon_kwargs) if own else None
        # 沒分到參數的 rank 也要有 param_groups，training loop 才能照常設定 lr
        self.param_groups = self.opt.param_groups if self.opt else [{"params": [], "lr": muon_kwargs["lr"]}]

    @torch.no_grad()
    def step(self):
        if self.opt:
            self.opt.step()
        works = [dist.broadcast(p.data, src=o, async_op=True) for p, o in zip(self.params, self.owner)]
        for w in works:
            w.wait()

    def zero_grad(self, set_to_none=True):
        # 非 owner 的參數也有 DDP 算出的梯度，要一起清
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_dict(self):
        return {"world": self.world, "rank": self.rank,
                "opt": self.opt.state_dict() if self.opt else None}

    def load_state_dict(self, sd):
        assert sd["world"] == self.world and sd["rank"] == self.rank, \
            f"sharded Muon state is for world {sd['world']} rank {sd['rank']}, not {self.world}/{self.rank}"
        if self.opt:
            self.opt.load_state_dict(sd["opt"])
