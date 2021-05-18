# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class PointerNet(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, vocabsize):
        super().__init__()
        self.linear = Linear(enc_hid_dim+dec_hid_dim,1)

    def forward(self, ctx, dec_hid):
        """
        dec_hid: bsz x tgtlen x hidsize
        c_tx : bsz x tgtlen x hidsize  
        """
        x = torch.cat((ctx,dec_hid), dim=-1)  ## bsz x tgtlen x hidsize ->  bsz x tgtlen x (hidsize x 2)
        x = self.linear(x)   ## bsz x tgtlen x 1
        x = torch.sigmoid(x) ## bsz x tgtlen x 1 
        return x

class ConsPosiEmb(nn.Module):
    """
    positional embeddings for constraints.
    """
    def __init__(self, embedding_dim, padding_idx, init_size=20, sep_id=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = ConsPosiEmb.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.positions = torch.tensor([])
        self.sep_id = sep_id
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_cons, embedding_dim, padding_idx=None, startpos=1025):
        """Build sinusoidal embeddings for constraints.
        input: num_cons, number of constraints. 
               embedding_dim, dimension of embeddings
               startpos: start position of constraints, to differentiate the position of constraint from the normal src words.
        """   
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(startpos, startpos+num_cons, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_cons, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_cons, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, startpos=1025):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = ConsPosiEmb.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
                startpos=startpos,
            )
        self.weights = self.weights.type_as(self._float_tensor)
        positions = self.get_positions(input, self.padding_idx, sep_id=self.sep_id)    
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def get_positions(self, tensor, padding_idx, startpos=0, tgt_tensor=None, sep_id=None):
        """Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. use constraint-right-pad-source=True
        padding_idx position=1, sep_idx position = 2 , others begin with 3, 
        a little different from the figure 2 in paper.
        """
        sep_cons = torch.ones_like(tensor)  
        bsz,clen = tensor.size()       
        for b in range(bsz):
            for j in range(clen):
                if tensor[b,j] == padding_idx:
                    break 
                sep_cons[b,j]= 2 if tensor[b,j] == sep_id else sep_cons[b,j-1]+1 
        return sep_cons
    
    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )
