import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import typing

import numpy as np

import os
import random
import warnings

warnings.filterwarnings("ignore")

class Fusion(layers.Layer):
    """Fusion layer supporting additive, convolutional, and cross-attention methods."""

    def __init__(self, method: str = "additive", name: str = "Fusion", **kwargs):
        super().__init__(name=name, **kwargs)
        self.method = method.lower()
        
        if self.method == "conv":
            self.convolution = tf.keras.layers.Conv1D(filters=1, kernel_size=1, name="conv_fusion")
        elif self.method == "cross_attention":
            # Define layers needed for cross-attention
            self.num_heads = 4  # Set number of heads based on projection dimensions
            self.projection_dim = 192  # Use the provided projection dimension for input
            self.q_proj = layers.Dense(self.projection_dim, name="q_proj")
            self.k_proj = layers.Dense(self.projection_dim, name="k_proj")
            self.v_proj = layers.Dense(self.projection_dim, name="v_proj")
            self.scale = (self.projection_dim // self.num_heads) ** -0.5
            self.output_proj = layers.Dense(self.projection_dim, name="output_proj")
        elif self.method== "concat":
            self.concat = layers.Concatenate(axis=-1)

    def call(self, x1, x2):

        if self.method == "conv":
            x = tf.stack([x1, x2], axis=-1)  # Stack along the last dimension
            x = self.convolution(x)  # Apply 1D convolution
            x = tf.squeeze(x, axis=-1)  # Remove the last dimension

        elif self.method == "cross_attention":
            # Cross-attention: Query from x1, Key and Value from x2
            batch_size = tf.shape(x1)[0]

            # Query from x1
            q =  self.q_proj(x1[:,0:1]) # self.q_proj(x1) 
            q = tf.reshape(q, (batch_size, -1, self.num_heads, self.projection_dim // self.num_heads))
            q = tf.transpose(q, perm=[0, 2, 1, 3])  # Shape: (B, num_heads, 8, head_dim)
            q = q * self.scale

            # Key and Value from x2
            k = self.k_proj(x2)
            k = tf.reshape(k, (batch_size, -1, self.num_heads, self.projection_dim // self.num_heads))
            k = tf.transpose(k, perm=[0, 2, 3, 1])  # Shape: (B, num_heads, head_dim, 8)

            v = self.v_proj(x2)
            v = tf.reshape(v, (batch_size, -1, self.num_heads, self.projection_dim // self.num_heads))
            v = tf.transpose(v, perm=[0, 2, 1, 3])  # Shape: (B, num_heads, 8, head_dim)

            # Compute attention
            attn_weights = tf.matmul(q, k)  # Shape: (B, num_heads, 8, 8)
            attn_weights = tf.nn.softmax(attn_weights, axis=-1)

            # Weighted sum of values
            attn_output = tf.matmul(attn_weights, v)  # Shape: (B, num_heads, 8, head_dim)
            attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # Shape: (B, 8, num_heads, head_dim)
            attn_output = tf.reshape(attn_output, (batch_size, -1, self.projection_dim))  # Shape: (B, 8, projection_dim)

            # Project the output back to the projection dimension
            x = self.output_proj(attn_output)
            x = tf.reduce_mean(x, axis=1)
        elif self.method== "concat":
            x = self.concat([x1, x2])
            # x = tf.stack([x1, x2], axis=-1)
            # x = tf.reduce_mean(x, axis=-1)
        elif self.method=="additive":
            x = tf.keras.layers.add([x1, x2])
        else:
            raise ("Fusion method must be either 'additive', 'concat', 'conv' or 'cross_attention'")
                    
        return x
    

class LayerScale(layers.Layer):
    """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239.

    Args:
        init_values (float): value to initialize the diagonal matrix of LayerScale.
        projection_dim (int): projection dimension used in LayerScale.
    """

    def __init__(self, init_values: float, projection_dim: int, name: str = "LayerScale", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = self.add_weight(
            shape=(projection_dim,),
            initializer=keras.initializers.Constant(init_values),
            name="gamma"
        )

    def call(self, x, training=False):
        return x * self.gamma


class StochasticDepth(layers.Layer):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382)."""

    def __init__(self, drop_prob: float, name: str = "StochasticDepth", **kwargs):
        super().__init__(name=name, **kwargs)
        self.drop_prob = drop_prob
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape, minval=0, maxval=1, seed=self.seed_generator
            )
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x



class ClassAttention(layers.Layer):
    """Class attention as proposed in CaiT: https://arxiv.org/abs/2103.17239."""

    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float, name: str = "ClassAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        head_dim = projection_dim // num_heads
        self.scale = head_dim**-0.5

        self.q = layers.Dense(projection_dim, name="q")
        self.k = layers.Dense(projection_dim, name="k")
        self.v = layers.Dense(projection_dim, name="v")
        self.attn_drop = layers.Dropout(dropout_rate, name="attn_drop")
        self.proj = layers.Dense(projection_dim, name="proj")
        self.proj_drop = layers.Dropout(dropout_rate, name="proj_drop")

    def call(self, x, training=False):
        batch_size, num_patches, num_channels = (
            tf.shape(x)[0],
            tf.shape(x)[1],
            tf.shape(x)[2],
        )

        q = tf.expand_dims(self.q(x[:, 0]), axis=1)
        q = tf.reshape(q, (batch_size, 1, self.num_heads, num_channels // self.num_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        scale = tf.cast(self.scale, dtype=q.dtype)
        q = q * scale

        k = self.k(x)
        k = tf.reshape(k, (batch_size, num_patches, self.num_heads, num_channels // self.num_heads))
        k = tf.transpose(k, perm=[0, 2, 3, 1])

        v = self.v(x)
        v = tf.reshape(v, (batch_size, num_patches, self.num_heads, num_channels // self.num_heads))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attn = tf.matmul(q, k)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x_cls = tf.matmul(attn, v)
        x_cls = tf.transpose(x_cls, perm=[0, 2, 1, 3])
        x_cls = tf.reshape(x_cls, (batch_size, 1, num_channels))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls, training=training)

        return x_cls, attn

class TalkingHeadAttention(layers.Layer):
    """Talking-head attention as proposed in CaiT: https://arxiv.org/abs/2003.02436."""

    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float, name: str = "TalkingHeadAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        head_dim = projection_dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = layers.Dense(projection_dim * 3, name="qkv")
        self.attn_drop = layers.Dropout(dropout_rate, name="attn_drop")
        self.proj = layers.Dense(projection_dim, name="proj")
        self.proj_l = layers.Dense(self.num_heads, name="proj_l")
        self.proj_w = layers.Dense(self.num_heads, name="proj_w")
        self.proj_drop = layers.Dropout(dropout_rate, name="proj_drop")

    def call(self, x, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads)) # 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = self.proj_l(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x, attn

def mlp(x, dropout_rate: float, hidden_units: typing.List[int], name: str = "MLP"):
    """FFN for a Transformer block."""
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation='gelu' if idx == 0 else None,
            bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
            name=f"{name}_dense_{idx}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"{name}_dropout_{idx}")(x)
    return x

def LayerScaleBlockClassAttention(
    projection_dim: int,
    num_heads: int,
    layer_norm_eps: float,
    init_values: float,
    mlp_units: typing.List[int],
    dropout_rate: float,
    sd_prob: float,
    name: str,
):
    """Pre-norm transformer block meant to be applied to the embeddings of the
    cls token and the embeddings of image patches.

    Includes LayerScale and Stochastic Depth.

    Args:
        projection_dim (int): projection dimension to be used in the
            Transformer blocks and patch projection layer.
        num_heads (int): number of attention heads.
        layer_norm_eps (float): epsilon to be used for Layer Normalization.
        init_values (float): initial value for the diagonal matrix used in LayerScale.
        mlp_units (List[int]): dimensions of the feed-forward network used in
            the Transformer blocks.
        dropout_rate (float): dropout rate to be used for dropout in the attention
            scores as well as the final projected outputs.
        sd_prob (float): stochastic depth rate.
        name (str): a name identifier for the block.

    Returns:
        A keras.Model instance.
    """
    x = keras.Input((None, projection_dim), name="x_input")
    x_cls = keras.Input((None, projection_dim), name="x_cls_input")
    inputs = keras.layers.Concatenate(axis=1, name="concatenate_inputs")([x_cls, x])

    # Class attention (CA).
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps, name="layer_norm")(inputs)
    attn_output, attn_scores = ClassAttention(projection_dim, num_heads, dropout_rate, name=f"{name}_class_attention")(
        x1
    )
    attn_output = (
        LayerScale(init_values, projection_dim, name=f"{name}_layer_scale")(attn_output)
        if init_values
        else attn_output
    )
    attn_output = StochasticDepth(sd_prob, name=f"{name}_stochastic_depth")(attn_output) if sd_prob else attn_output
    x2 = keras.layers.Add(name=f"{name}_add_1")([x_cls, attn_output])

    # FFN.
    x3 = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm_2")(x2)
    x4 = mlp(x3, hidden_units=mlp_units, dropout_rate=dropout_rate, name=f"{name}_mlp")
    x4 = LayerScale(init_values, projection_dim, name=f"{name}_layer_scale_2")(x4) if init_values else x4
    x4 = StochasticDepth(sd_prob, name=f"{name}_stochastic_depth_2")(x4) if sd_prob else x4
    outputs = keras.layers.Add(name=f"{name}_add_2")([x2, x4])

    return keras.Model([x, x_cls], [outputs, attn_scores], name=name)


def LayerScaleBlock(
    projection_dim: int,
    num_heads: int,
    layer_norm_eps: float,
    init_values: float,
    mlp_units: typing.List[int],
    dropout_rate: float,
    sd_prob: float,
    name: str,
):
    """Pre-norm transformer block meant to be applied to the embeddings of the
    image patches.

    Includes LayerScale and Stochastic Depth.

        Args:
            projection_dim (int): projection dimension to be used in the
                Transformer blocks and patch projection layer.
            num_heads (int): number of attention heads.
            layer_norm_eps (float): epsilon to be used for Layer Normalization.
            init_values (float): initial value for the diagonal matrix used in LayerScale.
            mlp_units (List[int]): dimensions of the feed-forward network used in
                the Transformer blocks.
            dropout_rate (float): dropout rate to be used for dropout in the attention
                scores as well as the final projected outputs.
            sd_prob (float): stochastic depth rate.
            name (str): a name identifier for the block.

    Returns:
        A keras.Model instance.
    """
    encoded_patches = keras.Input((None, projection_dim), name="encoded_patches")

    # Self-attention.
    x1 = layers.LayerNormalization(epsilon=layer_norm_eps, name="layer_norm")(encoded_patches)
    attn_output, attn_scores = TalkingHeadAttention(
        projection_dim, num_heads, dropout_rate, name=f"{name}_talking_head_attention"
    )(x1)
    attn_output = (
        LayerScale(init_values, projection_dim, name=f"{name}_layer_scale")(attn_output)
        if init_values
        else attn_output
    )
    attn_output = StochasticDepth(sd_prob, name=f"{name}_stochastic_depth")(attn_output) if sd_prob else attn_output
    x2 = layers.Add(name=f"{name}_add_1")([encoded_patches, attn_output])

    # FFN.
    x3 = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm_2")(x2)
    x4 = mlp(x3, hidden_units=mlp_units, dropout_rate=dropout_rate, name=f"{name}_mlp")
    x4 = LayerScale(init_values, projection_dim, name=f"{name}_layer_scale_2")(x4) if init_values else x4
    x4 = StochasticDepth(sd_prob, name=f"{name}_stochastic_depth_2")(x4) if sd_prob else x4
    outputs = layers.Add(name=f"{name}_add_2")([x2, x4])

    return keras.Model(encoded_patches, [outputs, attn_scores], name=name)


class FusionCaiT(keras.Model):
    """CaiT model."""

    def __init__(
        self,
        projection_dim: int,
        patch_size: int,
        num_patches: int,
        init_values: float,
        mlp_units: typing.List[int],
        sa_ffn_layers: int,
        ca_ffn_layers: int,
        num_heads: int,
        layer_norm_eps: float,
        dropout_rate: float,
        sd_prob: float,
        global_pool: str,
        pre_logits: bool,
        num_classes: int,
        method: str,
        **kwargs,
    ):
        if global_pool not in ["token", "avg"]:
            raise ValueError(
                'Invalid value received for `global_pool`, should be either `"token"` or `"avg"`.'
            )

        super().__init__(**kwargs)

        self.projection_ms = keras.Sequential(
            [
                layers.Conv2D(
                    filters=projection_dim,
                    kernel_size=(patch_size, patch_size),
                    strides=(patch_size, patch_size),
                    padding="VALID",
                    name="conv_projection_ms",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, projection_dim),
                    name="flatten_projection_ms",
                ),
            ],
            name="projection_ms",
        )

        self.projection_dem = keras.Sequential(
            [
                layers.Conv2D(
                    filters=projection_dim,
                    kernel_size=(patch_size, patch_size),
                    strides=(patch_size, patch_size),
                    padding="VALID",
                    name="conv_projection_dem",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, projection_dim),
                    name="flatten_projection_dem",
                ),
            ],
            name="projection_dem",
        )

        self.cls_token_ms = self.add_weight(shape=(1, 1, projection_dim), initializer="zeros", name="cls_token_ms")
        self.cls_token_dem = self.add_weight(shape=(1, 1, projection_dim), initializer="zeros", name="cls_token_dem")
        
        self.pos_embed_ms = self.add_weight(shape=(1, num_patches, projection_dim), initializer="zeros", name="pos_embed_ms")
        self.pos_embed_dem = self.add_weight(shape=(1, num_patches, projection_dim), initializer="zeros", name="pos_embed_dem")

        self.pos_drop_ms = layers.Dropout(dropout_rate, name="projection_dropout_ms")
        self.pos_drop_dem = layers.Dropout(dropout_rate, name="projection_dropout_dem")
        dpr = [sd_prob for _ in range(sa_ffn_layers)]

        self.blocks_ms = [
            LayerScaleBlock(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                sd_prob=dpr[i],
                name=f"ms_sa_ffn_block_{i}",
            )
            for i in range(sa_ffn_layers)
        ]
        self.blocks_dem = [
            LayerScaleBlock(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                sd_prob=dpr[i],
                name=f"dem_sa_ffn_block_{i}",
            )
            for i in range(sa_ffn_layers)
        ]

        self.blocks_token_only_ms = [
            LayerScaleBlockClassAttention(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                name=f"ms_ca_ffn_block_{i}",
                sd_prob=0.0,
            )
            for i in range(ca_ffn_layers)
        ]
        self.blocks_token_only_dem = [
            LayerScaleBlockClassAttention(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                name=f"dem_ca_ffn_block_{i}",
                sd_prob=0.0,
            )
            for i in range(ca_ffn_layers)
        ]

        self.fusion = Fusion(method=method, name=f"{method}FusionLayer")

        self.norm_ms = layers.LayerNormalization(epsilon=layer_norm_eps, name="ms_head_norm")
        self.norm_dem = layers.LayerNormalization(epsilon=layer_norm_eps, name="dem_head_norm")
        self.global_pool = global_pool
        self.pre_logits = pre_logits
        self.num_classes = num_classes
        if not pre_logits:
            self.head = layers.Dense(num_classes, activation="sigmoid", name="classification_head")

    def call(self, x, training=False):
        
        ms_x, dem_x = x

        ms_x = self.projection_ms(ms_x)
        ms_x = ms_x + self.pos_embed_ms
        ms_x = self.pos_drop_ms(ms_x)

        ms_sa_ffn_attn = {}
        for blk in self.blocks_ms:
            ms_x, ms_attn_scores = blk(ms_x)
            ms_sa_ffn_attn[f"ms_{blk.name}_att"] = ms_attn_scores

        ms_ca_ffn_attn = {}
        ms_cls_tokens = tf.tile(self.cls_token_ms, (tf.shape(dem_x)[0], 1, 1))
        for blk in self.blocks_token_only_ms:
            ms_cls_tokens, ms_attn_scores = blk([ms_x, ms_cls_tokens])
            ms_ca_ffn_attn[f"ms_{blk.name}_att"] = ms_attn_scores

        ms_x = tf.concat([ms_cls_tokens, ms_x], axis=1)
        ms_x = self.norm_ms(ms_x)

        if self.global_pool:
            ms_x = (
                tf.reduce_mean(ms_x[:, 1:], axis=1)
                if self.global_pool == "avg"
                else ms_x[:, 0]
            )


        dem_x = self.projection_dem(dem_x)
        dem_x = dem_x + self.pos_embed_dem
        dem_x = self.pos_drop_dem(dem_x)

        dem_sa_ffn_attn = {}
        for blk in self.blocks_dem:
            dem_x, dem_attn_scores = blk(dem_x)
            dem_sa_ffn_attn[f"dem_{blk.name}_att"] = dem_attn_scores

        dem_ca_ffn_attn = {}
        dem_cls_tokens = tf.tile(self.cls_token_dem, (tf.shape(dem_x)[0], 1, 1))
        for blk in self.blocks_token_only_dem:
            dem_cls_tokens, dem_attn_scores = blk([dem_x, dem_cls_tokens])
            dem_ca_ffn_attn[f"dem_{blk.name}_att"] = dem_attn_scores

        dem_x = tf.concat([dem_cls_tokens, dem_x], axis=1)
        dem_x = self.norm_dem(dem_x)

        if self.global_pool:
            dem_x = (
                tf.reduce_mean(dem_x[:, 1:], axis=1)
                if self.global_pool == "avg"
                else dem_x[:, 0]
            )

        # x = keras.layers.add([ms_x, dem_x])
        x = self.fusion(ms_x, dem_x)
        
        return (
            (x, ms_sa_ffn_attn, ms_ca_ffn_attn, dem_sa_ffn_attn, dem_ca_ffn_attn)
            if self.pre_logits
            else self.head(x)#, sa_ffn_attn, ca_ffn_attn)
        )

class CaiT(keras.Model):
    """CaiT model."""

    def __init__(
        self,
        projection_dim: int,
        patch_size: int,
        num_patches: int,
        init_values: float,
        mlp_units: typing.List[int],
        sa_ffn_layers: int,
        ca_ffn_layers: int,
        num_heads: int,
        layer_norm_eps: float,
        dropout_rate: float,
        sd_prob: float,
        global_pool: str,
        pre_logits: bool,
        num_classes: int,
        **kwargs,
    ):
        if global_pool not in ["token", "avg"]:
            raise ValueError(
                'Invalid value received for `global_pool`, should be either `"token"` or `"avg"`.'
            )

        super().__init__(**kwargs)

        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=projection_dim,
                    kernel_size=(patch_size, patch_size),
                    strides=(patch_size, patch_size),
                    padding="VALID",
                    name="conv_projection",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        self.cls_token = self.add_weight(
            shape=(1, 1, projection_dim), initializer="zeros"
        )
        self.pos_embed = self.add_weight(
            shape=(1, num_patches, projection_dim), initializer="zeros"
        )

        self.pos_drop = layers.Dropout(dropout_rate, name="projection_dropout")
        dpr = [sd_prob for _ in range(sa_ffn_layers)]

        self.blocks = [
            LayerScaleBlock(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                sd_prob=dpr[i],
                name=f"sa_ffn_block_{i}",
            )
            for i in range(sa_ffn_layers)
        ]

        self.blocks_token_only = [
            LayerScaleBlockClassAttention(
                projection_dim=projection_dim,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                init_values=init_values,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate,
                name=f"ca_ffn_block_{i}",
                sd_prob=0.0,
            )
            for i in range(ca_ffn_layers)
        ]

        self.norm = layers.LayerNormalization(epsilon=layer_norm_eps, name="head_norm")
        self.global_pool = global_pool
        self.pre_logits = pre_logits
        self.num_classes = num_classes
        if not pre_logits:
            self.head = layers.Dense(num_classes, activation="sigmoid", name="classification_head")

    def call(self, x, training=False):
        x = tf.ensure_shape(x, [None, 256, 256, 2]) 
        x = self.projection(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        sa_ffn_attn = {}
        for blk in self.blocks:
            x, attn_scores = blk(x)
            sa_ffn_attn[f"{blk.name}_att"] = attn_scores

        ca_ffn_attn = {}
        cls_tokens = tf.tile(self.cls_token, (tf.shape(x)[0], 1, 1))
        for blk in self.blocks_token_only:
            cls_tokens, attn_scores = blk([x, cls_tokens])
            ca_ffn_attn[f"{blk.name}_att"] = attn_scores

        x = tf.concat([cls_tokens, x], axis=1)
        x = self.norm(x)

        if self.global_pool:
            x = (
                tf.reduce_mean(x[:, 1:], axis=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        return (
            (x, sa_ffn_attn, ca_ffn_attn)
            if self.pre_logits
            else self.head(x)#, sa_ffn_attn, ca_ffn_attn)
        )


def get_config(
    image_size: int = 256,
    patch_size: int = 16,
    projection_dim: int = 192,
    sa_ffn_layers: int = 24,
    ca_ffn_layers: int = 2,
    num_heads: int = 4,
    mlp_ratio: int = 4,
    layer_norm_eps=1e-6,
    init_values: float = 1e-5,
    dropout_rate: float = 0.0,
    sd_prob: float = 0.0,
    global_pool: str = "token",
    pre_logits: bool = False,
    num_classes: int = 1,
    method: str = "cross_attention"
) -> typing.Dict:
    """Default configuration for CaiT models (cait_xxs24_224).

    Reference:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
    """
    config = {}

    # Patchification and projection.
    config["patch_size"] = patch_size
    config["num_patches"] = (image_size // patch_size) ** 2

    # LayerScale.
    config["init_values"] = init_values

    # Dropout and Stochastic Depth.
    config["dropout_rate"] = dropout_rate
    config["sd_prob"] = sd_prob

    # Shared across different blocks and layers.
    config["layer_norm_eps"] = layer_norm_eps
    config["projection_dim"] = projection_dim
    config["mlp_units"] = [
        projection_dim * mlp_ratio,
        projection_dim,
    ]

    # Attention layers.
    config["num_heads"] = num_heads
    config["sa_ffn_layers"] = sa_ffn_layers
    config["ca_ffn_layers"] = ca_ffn_layers

    # Representation pooling and task specific parameters.
    config["global_pool"] = global_pool
    config["pre_logits"] = pre_logits
    config["num_classes"] = num_classes

    # Fusion layer 
    config["method"] = method

    return config
