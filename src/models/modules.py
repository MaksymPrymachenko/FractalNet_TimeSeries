import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    """Conv - Dropout - BN - ReLU"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=None,
        pad_type="zero",
        dropout_pos="CDBR",
    ):
        """Conv
        Args:
            - dropout_pos: the position of dropout
                - CDBR (default): conv-dropout-BN-relu
                - CBRD: conv-BN-relu-dropout
                - FD: fractal-dropout
        """
        super().__init__()
        self.dropout_pos = dropout_pos
        if pad_type == "zero":
            self.pad = nn.ConstantPad1d(padding, 0)
        elif pad_type == "reflect":
            # Note: Reflection padding for 1D
            self.pad = nn.ReflectionPad1d(padding)
        else:
            raise ValueError(pad_type)

        self.conv = nn.Conv1d(C_in, C_out, kernel_size, stride, padding=0, bias=False)
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.bn = nn.BatchNorm1d(C_out)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if self.dropout_pos == "CDBR" and self.dropout:
            out = self.dropout(out)
        out = self.bn(out)
        out = F.relu_(out)
        if self.dropout_pos == "CBRD" and self.dropout:
            out = self.dropout(out)

        return out


class FractalBlock(nn.Module):
    def __init__(
        self,
        n_columns,
        C_in,
        C_out,
        p_ldrop,
        p_dropout,
        pad_type="zero",
        doubling=False,
        dropout_pos="CDBR",
    ):
        """Fractal block adapted for 1D data"""
        super().__init__()

        self.n_columns = n_columns
        self.p_ldrop = p_ldrop
        self.dropout_pos = dropout_pos
        if dropout_pos == "FD" and p_dropout > 0.0:
            self.dropout = nn.Dropout(p=p_dropout)
            p_dropout = 0.0
        else:
            self.dropout = None

        if doubling:
            self.doubler = ConvBlock(C_in, C_out, kernel_size=1, padding=0)
        else:
            self.doubler = None

        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns - 1)

        dist = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    first_block = i + 1 == dist
                    if first_block and not doubling:
                        cur_C_in = C_in
                    else:
                        cur_C_in = C_out

                    module = ConvBlock(
                        cur_C_in,
                        C_out,
                        dropout=p_dropout,
                        pad_type=pad_type,
                        dropout_pos=dropout_pos,
                    )
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

    def drop_mask(self, B, global_cols, n_cols):
        """Generate drop mask; [n_cols, B]."""
        GB = global_cols.shape[0]
        gdrop_cols = global_cols - (self.n_columns - n_cols)
        gdrop_indices = np.where(gdrop_cols >= 0)[0]
        gdrop_mask = np.zeros([n_cols, GB], dtype=np.float32)
        gdrop_mask[gdrop_cols[gdrop_indices], gdrop_indices] = 1.0

        LB = B - GB
        ldrop_mask = np.random.binomial(1, 1.0 - self.p_ldrop, [n_cols, LB]).astype(
            np.float32
        )
        alive_count = ldrop_mask.sum(axis=0)
        dead_indices = np.where(alive_count == 0.0)[0]
        ldrop_mask[
            np.random.randint(0, n_cols, size=dead_indices.shape), dead_indices
        ] = 1.0

        drop_mask = np.concatenate((gdrop_mask, ldrop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    def join(self, outs, global_cols):
        n_cols = len(outs)
        out = torch.stack(outs)  # [n_cols, B, C, L]
        if self.training:
            mask = self.drop_mask(out.size(1), global_cols, n_cols).to(
                out.device
            )  # [n_cols, B]
            mask = mask.view(*mask.size(), 1, 1)  # unsqueeze to [n_cols, B, 1, 1]
            n_alive = mask.sum(dim=0)  # [B, 1, 1]
            masked_out = out * mask  # [n_cols, B, C, L]
            n_alive[n_alive == 0.0] = 1.0
            out = masked_out.sum(dim=0) / n_alive  # [B, C, L]
        else:
            out = out.mean(dim=0)

        return out

    def forward(self, x, global_cols, deepest=False):
        out = self.doubler(x) if self.doubler else x
        outs = [out] * self.n_columns
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_outs = []
            if deepest:
                st = self.n_columns - 1

            for c in range(st, self.n_columns):
                cur_in = outs[c]
                cur_module = self.columns[c][i]
                cur_outs.append(cur_module(cur_in))

            joined = self.join(cur_outs, global_cols)

            for c in range(st, self.n_columns):
                outs[c] = joined

        if self.dropout_pos == "FD" and self.dropout:
            outs[-1] = self.dropout(outs[-1])

        return outs[-1]
