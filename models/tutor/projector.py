import torch.nn as nn

class StrategyProjector(nn.Module):
    def __init__(self, llm_dim=4096, trm_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, trm_dim)
        )

    def forward(self, x):
        # output is reshaped to match the expected puzzle_emb dimensions: [Batch, 1, 512]
        out = self.net(x)
        return out.unsqueeze(1)