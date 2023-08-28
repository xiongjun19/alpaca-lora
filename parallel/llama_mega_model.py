import math
import torch
from torch import nn
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.module import MegatronModule

from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.core import mpu
from megatron.arguments import core_transformer_config_from_args
from megatron import get_num_microbatches


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    if labels is None:
        # [s b h] => [b s h]
        return lm_output
    else:
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        output = lm_output.transpose(0, 1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class Embedding(MegatronModule):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        embedding_weights_in_fp32: casts word embedding weights to
                                   fp32 before sampling. Required to
                                   maintain reproducibility when
                                   training in bf16.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 embedding_weights_in_fp32=False):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        args = get_args()
        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config,
            init_method=lambda x: x)
        self._word_embeddings_key = 'word_embeddings'
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # Dropout.
        embeddings = self.embedding_dropout(embeddings)
        return embeddings


class LlamaModel(MegatronModule):
    def __init__(self,
                 config,
                 mega_config,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super().init(config=config, share_embeddings_and_output_weights=False)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.encoder_hidden_state = None
        self.emb_dropout = 0.1
        self.max_seq_len = 2048
        args = get_args()

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       self.vocab_size,
                                       self.max_seq_len,
                                       self.emb_dropout,
                                       mega_config,
                                       args.embedding_weights_in_fp32)
            self._embedding_key = 'embedding'
        self.num_layers = _get_num_layers(args, config.num_hidden_layers)
        offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
        self.layers = nn.ModuleList(
            [ParallelLlamaLayer(config, mega_config, i + offset)
                for i in range(self.num_layers)])
        if self.post_process:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.out = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.vocab_size,
                mega_config, lambda x: x,
                bias=False, gather_output=True)


        self.gradient_checkpointing = False
        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0

        # self.post_init()

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def get_input_tensor(self):
        return self.input_tensor

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(self, input_ids, attention_mask, position_ids, labels=None):
        if self.pre_process:
            encoder_input = self.embedding(input_ids, position_ids,
                                           tokentype_ids=None)
        else:
            encoder_input = self.input_tensor
        # Determine if the current iteration is first microbatch
        if self.num_microbatches_in_previous_step != get_num_microbatches():
            # Reset count on new batch size rampup interval
            self.microbatch_count = 0
        self.num_microbatches_in_previous_step = get_num_microbatches()
        for index in range(self.num_layers):
            layer = self.layers(index)
            encoder_input = layer(encoder_input, attention_mask, position_ids)
        # Skip counter update for eval and activation checkpointing
        if torch.is_grad_enabled() and self.training:
            self.microbatch_count += 1
        res = encoder_input
        if self.post_process:
            res = self.final_layernorm(res)
            res = self.out(res)
        return res


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _get_num_layers(args, cfg_layers):
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        assert cfg_layers % args.transformer_pipeline_model_parallel_size == 0, \
            'num_layers must be divisible by transformer_pipeline_model_parallel_size'

        # when a standalone embedding stage is used, all transformer layers
        # are divided among pipeline rank >= 1, while on pipeline rank 0,
        # ranks either contain the input embedding layer (virtual pp rank 0),
        # or no layers at all (virtual pp rank >= 1).
        num_layers = (
            0
            if args.standalone_embedding_stage
            and mpu.get_pipeline_model_parallel_rank() == 0 else
            args.num_layers // args.transformer_pipeline_model_parallel_size
        )
        return num_layers
    else:
        return cfg_layers


class ParallelLlamaLayer(MegatronModule):
    def __init__(self, config, mega_config, layer_number):
        super().__init__()
        args = get_args()
        self.mega_cfg = mega_config
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size
        self.self_attn = ParaLlamaAttention(config, mega_config)
        self.mlp = ParaLlamaMLP(config, mega_config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            position_ids:

        """
        residual = hidden_states
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        hidden_states = \
            self.self_attn(layernorm_output, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ParaLlamaAttention(MegatronModule):
    def __init__(self, config, mega_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.mega_config = mega_config

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.q_proj = tensor_parallel.ColumnParallelLinear(self.hidden_size,
                self.num_heads * self.head_dim,
                self.mega_config, lambda x: x,
                bias=False, gather_output=False)
        self.k_proj = tensor_parallel.ColumnParallelLinear(self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.mega_config, lambda x: x,
                bias=False, gather_output=False)
        self.v_proj = tensor_parallel.ColumnParallelLinear(self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                self.mega_config, lambda x: x,
                bias=False, gather_output=False)

        self.wo = tensor_parallel.RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            config=self.mega_config,
            init_method=lambda x: x,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states, _ = self.q_proj(hidden_states)
        key_states, _ = self.k_proj(hidden_states)
        value_states, _ = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads / self.tp_size, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads / self.tp_size, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads / self.tp_size, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads / self.tp_size, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads / self.tp_size, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output, _ = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ParaLlamaMLP(MegatronModule):
    def __init__(self, config, mega_config):
        super().__init__()
        args = get_args()
        self.add_bias = False
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.functional.silu
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.mega_config = mega_config

        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            config=self.mega_config,
            init_method=lambda x: x,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True
        )

        self.up_proj = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            config=self.mega_config,
            init_method=lambda x: x,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True
        )

        # Project back to h.
        self.down_proj = tensor_parallel.RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            config=self.mega_config,
            init_method=lambda x: x,
            bias=self.add_bias,
            input_is_parallel=True
        )

    def forward(self, x):
        gate_out, _ = self.gate_proj(x)
        up_out, _ = self.up_proj(x)
        res, _ = self.down_proj(self.act_fn(gate_out) * up_out)
        return res

