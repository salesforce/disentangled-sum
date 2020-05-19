# Copyright (c) 2018, salesforce.com, inc.

# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""Seq2seq model using BART."""

import copy
import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_bart import (
    _filter_out_falsey_values,
    _make_linear_from_emb,
    _prepare_bart_decoder_inputs,
    _reorder_buffer,
    BartConfig,
    BartDecoder,
    BartEncoder,
    BartForConditionalGeneration,
    BartModel,
    invert_mask,
    LayerNorm,
    LearnedPositionalEmbedding,
    PretrainedBartModel,
    SelfAttention,
    SinusoidalPositionalEmbedding,
)


def validate_tensor(loss):
    if torch.isinf(loss) or torch.isnan(loss):
        import pdb

        pdb.set_trace()
        return False
    return True


class DecoderLayerLM(nn.Module):
    """Transformer decoder that can operate without encoder_hidden_states"""

    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        # Disable this if encoder_states is not fed
        if encoder_hidden_states is not None:
            residual = x
            assert self.encoder_attn.cache_key != self.self_attn.cache_key
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            x, _ = self.encoder_attn(
                query=x,
                key=encoder_hidden_states,
                key_padding_mask=encoder_attn_mask,
                layer_state=layer_state,  # mutates layer state
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoderLM(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayerLM`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.d_model, config.pad_token_id
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.d_model, self.padding_idx
            )
        self.layers = nn.ModuleList(
            [DecoderLayerLM(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = (
            LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        )
        self.layer_norm = (
            LayerNorm(config.d_model) if config.add_final_layer_norm else None
        )

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = (
                decoder_cached_states[idx]
                if decoder_cached_states is not None
                else None
            )

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [
            hidden_state.transpose(0, 1) for hidden_state in all_hidden_states
        ]
        x = x.transpose(0, 1)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = (
                (encoder_hidden_states, encoder_padding_mask),
                next_decoder_cache,
            )
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class BartModelLM(BartModel):
    """BartModel but the decoder is BartDecoderLM."""

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.decoder = BartDecoderLM(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
    ):

        # make masks if user doesn't supply
        if not use_cache:
            (
                decoder_input_ids,
                decoder_padding_mask,
                causal_mask,
            ) = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if input_ids is None and encoder_outputs is None:
            encoder_outputs = (None, None)
        elif encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs


class BartForConditionalGenerationLM(BartForConditionalGeneration):
    """Bart for conditional generation using the BartModelLM."""

    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelLM(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )


class MultiHeadBartDecoder(BartDecoder):
    """BartDecoder with separate decoder layers at the final level."""

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__(config, embed_tokens)
        # override
        self.layers = nn.ModuleList(
            [DecoderLayerLM(config) for _ in range(config.decoder_layers - 1)]
        )  # type: List[DecoderLayer]
        self.final_layers = nn.ModuleList(
            [DecoderLayerLM(config) for _ in range(config.final_layers)]
        )

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        use_cache=False,
        final_layer=None,
        **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation
            final_layer (int): the final layer ID to switch when necessary (decoding).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = (
                decoder_cached_states[idx]
                if decoder_cached_states is not None
                else None
            )

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # final layer
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if self.output_hidden_states:
            all_hidden_states += (x,)
        dropout_probability = random.uniform(0, 1)
        if not (self.training and (dropout_probability < self.layerdrop)):
            layer_state = (
                decoder_cached_states[-1] if decoder_cached_states is not None else None
            )

            x, layer_self_attn, layer_past = self.final_layers[final_layer](
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm:  # last layer of mbart
                x = self.layer_norm(x)

            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [h.transpose(0, 1) for h in all_hidden_states]
        x = x.transpose(0, 1)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        if use_cache:
            next_cache = (
                (encoder_hidden_states, encoder_padding_mask),
                next_decoder_cache,
            )
        else:
            next_cache = None

        return x, next_cache, all_hidden_states, list(all_self_attns)


class MultiHeadBartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = MultiHeadBartDecoder(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        use_cache=False,
        final_layer=None,
    ):
        if encoder_outputs is None and input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        elif encoder_outputs is None:
            encoder_outputs = (None,)
        assert isinstance(encoder_outputs, tuple)

        if decoder_cached_states is None:
            decoder_cached_states = [None] * len(decoder_input_ids)

        if decoder_attention_mask is None:
            decoder_attention_mask = [None] * len(decoder_input_ids)

        all_dec_outputs = []
        if isinstance(final_layer, int):
            if isinstance(decoder_input_ids, list):
                decoder_input_ids = [decoder_input_ids[final_layer]]
                decoder_attention_mask = [decoder_attention_mask[final_layer]]
                decoder_cached_states = [decoder_cached_states[final_layer]]
            else:  # decoder doesn't come in multi output: generation time
                decoder_input_ids = [decoder_input_ids]
                decoder_attention_mask = [decoder_attention_mask]
                decoder_cached_states = (
                    decoder_cached_states
                    if decoder_cached_states[0] is None
                    else [decoder_cached_states]
                )

        # If final_layer is None (i.e. at training time), it will compute both outputs
        # Otherwise a specified decoder branch will be used
        for idx, (d_input_ids, d_attn_mask, d_cached_states) in enumerate(
            zip(decoder_input_ids, decoder_attention_mask, decoder_cached_states)
        ):
            # make masks if user doesn't supply
            if not use_cache:
                (
                    d_input_ids,
                    d_padding_mask,
                    causal_mask,
                ) = _prepare_bart_decoder_inputs(
                    self.config,
                    input_ids,
                    decoder_input_ids=d_input_ids,
                    decoder_padding_mask=d_attn_mask,
                    causal_mask_dtype=self.shared.weight.dtype,
                )
            else:
                d_padding_mask, causal_mask = None, None

            assert decoder_input_ids is not None
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                d_input_ids,
                encoder_outputs[0],
                attention_mask,
                d_padding_mask,
                decoder_causal_mask=causal_mask,
                decoder_cached_states=d_cached_states,
                use_cache=use_cache,
                final_layer=(final_layer if final_layer is not None else idx),
            )
            all_dec_outputs.append(decoder_outputs)

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs = [
            _filter_out_falsey_values(d_outs) for d_outs in all_dec_outputs
        ]
        assert isinstance(decoder_outputs[0][0], torch.Tensor)
        encoder_outputs = _filter_out_falsey_values(encoder_outputs)
        return (decoder_outputs, encoder_outputs)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


class MultiHeadBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = MultiHeadBartModel(config)
        self.model = base_model
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=False,
        final_layer=None,
        **unused,
    ):
        dec_outputs, enc_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            final_layer=final_layer,
        )
        all_outputs = []
        # iterate over multiple (final-)layer outputs
        for idx, output in enumerate(dec_outputs):
            lm_logits = F.linear(
                output[0], self.model.shared.weight, bias=self.final_logits_bias
            )
            # Add cache, hidden states and attention if they are here
            outputs = (lm_logits,) + output[1:] + enc_outputs
            if lm_labels is not None:
                assert len(lm_labels) == len(dec_outputs)
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size),
                    lm_labels[idx].reshape(-1),
                )
                assert validate_tensor(masked_lm_loss)
                outputs = (masked_lm_loss,) + outputs
            all_outputs.append(outputs)

        # specific expert
        if isinstance(final_layer, int) and final_layer < len(all_outputs):
            all_outputs = all_outputs[final_layer]
        elif isinstance(final_layer, int):
            all_outputs = all_outputs[0]

        return all_outputs

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, **kwargs
    ):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "final_layer": kwargs["final_layer"],
        }

    @classmethod
    def from_pretrained_multi(
        cls, full_model_name_or_path, full_model_config=None, final_layers=2
    ):
        """Load either a full seq2seq model, or pre-load model from
        a separate encoder and decoder stacks."""

        bart_config = BartConfig.from_pretrained(full_model_name_or_path)
        bart_config.final_layers = final_layers
        # initialize with random weights
        model = cls(bart_config)

        # this is actually a complete checkpoint
        if os.path.exists(full_model_name_or_path):
            ckpt = torch.load(
                os.path.join(full_model_name_or_path, "pytorch_model.bin"),
                map_location="cpu",
            )
            model.load_state_dict(ckpt)
            return model

        # initialize scratch
        bart_model = BartModel.from_pretrained(full_model_name_or_path)

        # encoder full copy
        model.model.encoder.load_state_dict(bart_model.encoder.state_dict())
        model.model.decoder.embed_tokens.load_state_dict(
            bart_model.decoder.embed_tokens.state_dict()
        )
        model.model.decoder.embed_positions.load_state_dict(
            bart_model.decoder.embed_positions.state_dict()
        )

        # excluding the last layer
        for ml, bl in zip(model.model.decoder.layers, bart_model.decoder.layers):
            ml.load_state_dict(bl.state_dict())

        for ml in model.model.decoder.final_layers:
            ml.load_state_dict(bart_model.decoder.layers[-1].state_dict())

        del bart_model

        return model


class MultiInputBartForConditionalGeneration(BartForConditionalGenerationLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=None,
        informativeness=False,
        **unused,
    ):
        """Unlike the normal forward function, input_ids can be a list of those,
        representing multiple input sources.
        """
        # evaluate, or only processing the paper
        if input_ids is None or isinstance(input_ids, torch.Tensor):
            return super().forward(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
            )

        if attention_mask is not None:
            assert len(input_ids) == len(attention_mask)
        else:
            attention_mask = [None] * len(input_ids)

        # calculate target lm logprob : p(y) using the decoder at training time
        if lm_labels is not None and not informativeness:
            # run LM on the target
            target_lm_outputs = self.model(
                input_ids=None,  # setting this to none will skip cross-attention
                attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
            )
            target_lm_logits = F.linear(
                target_lm_outputs[0],
                self.model.shared.weight,
                bias=self.final_logits_bias,
            )
            logp_target = -F.cross_entropy(
                target_lm_logits.view(-1, self.config.vocab_size),
                lm_labels.view(-1),
                reduction="none",
            )
            logp_target = torch.mean(logp_target[lm_labels.view(-1) != -100])

        losses = []
        for idx, (i_ids, a_mask) in enumerate(zip(input_ids, attention_mask)):
            # From huggingface/transformers/src/transformers/modeling_bart.py
            outputs = self.model(
                i_ids,
                attention_mask=a_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
            )
            lm_logits = F.linear(
                outputs[0], self.model.shared.weight, bias=self.final_logits_bias
            )
            # Add cache, hidden states and attention if they are here
            outputs = (lm_logits,) + outputs[1:]
            if lm_labels is not None:
                # cross entropy over against summary
                if idx == 0:
                    masked_lm_loss = F.cross_entropy(
                        lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1)
                    )
                    assert validate_tensor(masked_lm_loss)
                    outputs = (masked_lm_loss,) + outputs

                # additional loss
                else:
                    # logp
                    logp = -F.cross_entropy(
                        lm_logits.view(-1, self.config.vocab_size),
                        lm_labels.view(-1),
                        reduction="none",
                    )
                    logp = torch.mean(logp[lm_labels.view(-1) != -100])
                    if informativeness:
                        # masked_lm_loss * log p
                        info = torch.exp(-losses[0]) * logp
                        mutual_info = info
                    else:
                        mutual_info = torch.exp(logp) * (logp - logp_target)
                    assert validate_tensor(mutual_info)
                    outputs = (mutual_info,) + outputs

            losses.append(outputs[0])

        return (losses,)


class MultiInputMultiHeadBartForConditionalGeneration(
    MultiHeadBartForConditionalGeneration
):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        use_cache=None,
        input_modes=None,
        final_layer=None,
        informativeness=False,
        **unused,
    ):
        """Unlike the normal forward function, input_ids can be a list of those,
        representing multiple input sources.

        input_modes specify the list of objectives. (LogL, MI_inbound, MI_outbound)
        """
        # evaluate, or only processing the paper; fallback to normal MultiHead
        if input_ids is None or isinstance(input_ids, torch.Tensor):
            assert final_layer is None or isinstance(final_layer, int)
            return super().forward(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
                final_layer=final_layer,
            )

        if attention_mask is not None:
            assert len(input_ids) == len(attention_mask)
            assert len(input_ids) == len(final_layer)
            assert len(input_ids) == len(input_modes)
        else:
            attention_mask = [None] * len(input_ids)

        logp_targets = []
        if lm_labels is not None and not informativeness:  # training time
            # run LM on the target
            target_lm_outputs, _ = self.model(
                input_ids=None,  # setting this to none will skip cross-attention
                attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
                final_layer=None,
            )
            for d_idx, output in enumerate(target_lm_outputs):
                target_lm_logits = F.linear(
                    output[0], self.model.shared.weight, bias=self.final_logits_bias
                )
                logp_target = -F.cross_entropy(
                    target_lm_logits.view(-1, self.config.vocab_size),
                    lm_labels[d_idx].reshape(-1),
                    reduction="none",
                )
                logp_target = torch.mean(
                    logp_target[lm_labels[d_idx].reshape(-1) != -100]
                )
                logp_targets.append(logp_target)

        # loop over input sources
        losses = {}
        for idx, (i_ids, a_mask, f_layer, mode) in enumerate(
            zip(input_ids, attention_mask, final_layer, input_modes)
        ):

            # dec_outputs is a list, accompany decoding results for (contrib, context)
            # in this order, if final_layer is not specified.
            dec_outputs, enc_outputs = self.model(
                i_ids,
                attention_mask=a_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
                final_layer=f_layer,
            )

            # Main CE loss (against both contrib and context summaries)
            if mode == "LogL":
                all_outputs = []
                # iterate over multiple (final-)layer outputs
                for d_idx, output in enumerate(dec_outputs):
                    lm_logits = F.linear(
                        output[0], self.model.shared.weight, bias=self.final_logits_bias
                    )
                    # Add cache, hidden states and attention if they are here
                    outputs = (lm_logits,) + output[1:] + enc_outputs
                    if lm_labels is not None:
                        assert len(lm_labels) == len(dec_outputs)
                        loss_fct = nn.CrossEntropyLoss()
                        masked_lm_loss = loss_fct(
                            lm_logits.view(-1, self.config.vocab_size),
                            lm_labels[d_idx].reshape(-1),
                        )
                        assert validate_tensor(masked_lm_loss)
                        outputs = (masked_lm_loss,) + outputs
                    all_outputs.append(outputs)

                losses[mode] = all_outputs

            # auxiliary losses (either MI or informativeness)
            elif mode.startswith("MI"):
                # in this case, dec_outputs is not a list
                # assert isinstance(f_layer, int)
                all_outputs = []
                # iterate over multiple (final-)layer outputs
                for d_idx, output in enumerate(dec_outputs):
                    lm_logits = F.linear(
                        output[0], self.model.shared.weight, bias=self.final_logits_bias
                    )
                    # Add cache, hidden states and attention if they are here
                    outputs = (lm_logits,) + output[1:] + enc_outputs
                    if lm_labels is not None:
                        labels = lm_labels[d_idx]
                        # logp
                        logp = -F.cross_entropy(
                            lm_logits.view(-1, self.config.vocab_size),
                            labels.reshape(-1),
                            reduction="none",
                        )
                        logp = torch.mean(logp[lm_labels[d_idx].reshape(-1) != -100])
                        if informativeness:
                            info = torch.exp(-losses["LogL"][0][0]) * logp
                            mutual_info = info
                        else:
                            mutual_info = torch.exp(logp) * (logp - logp_target)
                        assert validate_tensor(mutual_info)
                        outputs = (mutual_info,) + outputs

                    all_outputs.append(outputs)

                losses[mode] = all_outputs

        return losses
