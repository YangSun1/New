FeatureDecoder(
  (segmentation_model): ModuleList(
    (0): DINOv3_Adapter(
      (backbone): DinoVisionTransformer(
        (patch_embed): PatchEmbed(
          (proj): Conv2d(3, 4096, kernel_size=(16, 16), stride=(16, 16))
          (norm): Identity()
        )
        (rope_embed): RopePositionEmbedding()
        (blocks): ModuleList(
          (0-39): 40 x SelfAttentionBlock(
            (norm1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (attn): SelfAttention(
              (qkv): LinearKMaskedBias(in_features=4096, out_features=12288, bias=False)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=4096, out_features=4096, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): LayerScale()
            (norm2): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (mlp): SwiGLUFFN(
              (w1): Linear(in_features=4096, out_features=8192, bias=True)
              (w2): Linear(in_features=4096, out_features=8192, bias=True)
              (w3): Linear(in_features=8192, out_features=4096, bias=True)
            )
            (ls2): LayerScale()
          )
        )
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (local_cls_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (head): Identity()
      )
      (spm): SpatialPriorModule(
        (stem): Sequential(
          (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (4): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (7): SyncBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
          (9): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (conv2): Sequential(
          (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): SyncBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (conv3): Sequential(
          (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (conv4): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (fc1): Conv2d(64, 4096, kernel_size=(1, 1), stride=(1, 1))
        (fc2): Conv2d(128, 4096, kernel_size=(1, 1), stride=(1, 1))
        (fc3): Conv2d(256, 4096, kernel_size=(1, 1), stride=(1, 1))
        (fc4): Conv2d(256, 4096, kernel_size=(1, 1), stride=(1, 1))
      )
      (interactions): Sequential(
        (0): InteractionBlockWithCls(
          (extractor): Extractor(
            (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
              (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
              (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
              (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
            )
            (ffn): ConvFFN(
              (fc1): Linear(in_features=4096, out_features=1024, bias=True)
              (dwconv): DWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=4096, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (drop_path): DropPath()
          )
        )
        (1): InteractionBlockWithCls(
          (extractor): Extractor(
            (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
              (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
              (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
              (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
            )
            (ffn): ConvFFN(
              (fc1): Linear(in_features=4096, out_features=1024, bias=True)
              (dwconv): DWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=4096, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (drop_path): DropPath()
          )
        )
        (2): InteractionBlockWithCls(
          (extractor): Extractor(
            (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
              (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
              (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
              (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
            )
            (ffn): ConvFFN(
              (fc1): Linear(in_features=4096, out_features=1024, bias=True)
              (dwconv): DWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=4096, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (drop_path): DropPath()
          )
        )
        (3): InteractionBlockWithCls(
          (extractor): Extractor(
            (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
              (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
              (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
              (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
            )
            (ffn): ConvFFN(
              (fc1): Linear(in_features=4096, out_features=1024, bias=True)
              (dwconv): DWConv(
                (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
              )
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=4096, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
            (drop_path): DropPath()
          )
          (extra_extractors): Sequential(
            (0): Extractor(
              (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (attn): MSDeformAttn(
                (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
                (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
                (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
                (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
              )
              (ffn): ConvFFN(
                (fc1): Linear(in_features=4096, out_features=1024, bias=True)
                (dwconv): DWConv(
                  (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                )
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=4096, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
              (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (drop_path): DropPath()
            )
            (1): Extractor(
              (query_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (feat_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (attn): MSDeformAttn(
                (sampling_offsets): Linear(in_features=4096, out_features=128, bias=True)
                (attention_weights): Linear(in_features=4096, out_features=64, bias=True)
                (value_proj): Linear(in_features=4096, out_features=2048, bias=True)
                (output_proj): Linear(in_features=2048, out_features=4096, bias=True)
              )
              (ffn): ConvFFN(
                (fc1): Linear(in_features=4096, out_features=1024, bias=True)
                (dwconv): DWConv(
                  (dwconv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                )
                (act): GELU(approximate='none')
                (fc2): Linear(in_features=1024, out_features=4096, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
              (ffn_norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
              (drop_path): DropPath()
            )
          )
        )
      )
      (up): ConvTranspose2d(4096, 4096, kernel_size=(2, 2), stride=(2, 2))
      (norm1): SyncBatchNorm(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (norm2): SyncBatchNorm(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (norm3): SyncBatchNorm(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (norm4): SyncBatchNorm(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): Mask2FormerHead(
      (pixel_decoder): MSDeformAttnPixelDecoder(
        (input_convs): ModuleList(
          (0-2): 3 x Sequential(
            (0): Conv2d(4096, 2048, kernel_size=(1, 1), stride=(1, 1))
            (1): GroupNorm(32, 2048, eps=1e-05, affine=True)
          )
        )
        (encoder): MSDeformAttnTransformerEncoderOnly(
          (encoder): MSDeformAttnTransformerEncoder(
            (layers): ModuleList(
              (0-5): 6 x MSDeformAttnTransformerEncoderLayer(
                (self_attn): MSDeformAttn(
                  (sampling_offsets): Linear(in_features=2048, out_features=384, bias=True)
                  (attention_weights): Linear(in_features=2048, out_features=192, bias=True)
                  (value_proj): Linear(in_features=2048, out_features=2048, bias=True)
                  (output_proj): Linear(in_features=2048, out_features=2048, bias=True)
                )
                (dropout1): Dropout(p=0.0, inplace=False)
                (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
                (linear1): Linear(in_features=2048, out_features=4096, bias=True)
                (dropout2): Dropout(p=0.0, inplace=False)
                (linear2): Linear(in_features=4096, out_features=2048, bias=True)
                (dropout3): Dropout(p=0.0, inplace=False)
                (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
        )
        (pe_layer): Positional encoding PositionEmbeddingSine
            num_pos_feats: 1024
            temperature: 10000
            normalize: True
            scale: 6.283185307179586
        (mask_feature): Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1))
        (lateral_convs): ModuleList(
          (0): Conv2d(
            4096, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
            (norm): GroupNorm(32, 2048, eps=1e-05, affine=True)
          )
        )
        (output_convs): ModuleList(
          (0): Conv2d(
            2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            (norm): GroupNorm(32, 2048, eps=1e-05, affine=True)
          )
        )
      )
      (predictor): MultiScaleMaskedTransformerDecoder(
        (pe_layer): Positional encoding PositionEmbeddingSine
            num_pos_feats: 1024
            temperature: 10000
            normalize: True
            scale: 6.283185307179586
        (transformer_self_attention_layers): ModuleList(
          (0-8): 9 x SelfAttentionLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=2048, out_features=2048, bias=True)
            )
            (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (transformer_cross_attention_layers): ModuleList(
          (0-8): 9 x CrossAttentionLayer(
            (multihead_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=2048, out_features=2048, bias=True)
            )
            (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (transformer_ffn_layers): ModuleList(
          (0-8): 9 x FFNLayer(
            (linear1): Linear(in_features=2048, out_features=4096, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (linear2): Linear(in_features=4096, out_features=2048, bias=True)
            (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
        )
        (post_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (query_feat): Embedding(100, 2048)
        (query_embed): Embedding(100, 2048)
        (level_embed): Embedding(3, 2048)
        (input_proj): ModuleList(
          (0-2): 3 x Sequential()
        )
        (class_embed): Linear(in_features=2048, out_features=151, bias=True)
        (mask_embed): MLP(
          (layers): ModuleList(
            (0-2): 3 x Linear(in_features=2048, out_features=2048, bias=True)
          )
        )
      )
    )
  )
)