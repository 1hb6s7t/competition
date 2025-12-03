# MoE模型优化技术报告

## 📊 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 92.6471 |
| Prefill时延得分 | 599.2163 |
| Decode时延得分 | 262.4934 |
| **总分** | **318.1189** |

## 🎯 优化模型

本项目针对以下两个MoE（Mixture of Experts）模型进行了昇腾NPU适配与性能优化：

1. **DeepSeek-MoE-16B-Chat** - 深度求索开源的MoE大模型
2. **Qwen1.5-MoE-A2.7B-Chat** - 通义千问开源的MoE大模型

---

## 🔧 核心优化技术

### 1. MindSpore算子适配优化

#### 1.1 使用mint算子替代ops算子

将`ops`算子替换为`mint`算子：

```python
# 优化前
ops.arange(0, self.dim, 2)
ops.cat((freqs, freqs), dim=-1)
ops.topk(routing_weights, top_k, dim=-1)

# 优化后
mint.arange(0, self.dim, 2, dtype=mindspore.int64)
mint.cat((freqs, freqs), dim=-1)
mint.topk(routing_weights, self.top_k, dim=-1)
```

**收益**：减少了算子下发前的判断。

#### 1.2 使用F.embedding替代索引操作

```python
# 优化前
cos = cos[position_ids].unsqueeze(unsqueeze_dim)
sin = sin[position_ids].unsqueeze(unsqueeze_dim)

# 优化后
cos = F.embedding(position_ids, cos).unsqueeze(unsqueeze_dim)
sin = F.embedding(position_ids, sin).unsqueeze(unsqueeze_dim)
```

**收益**：避免动态索引带来的性能损失，提升RoPE位置编码的计算效率。

#### 1.3 使用mint.narrow替代切片操作

```python
# 优化前
self.cos_cached[:seq_len].to(dtype=x.dtype)
position_ids = position_ids[:, -input_ids.shape[1]:]

# 优化后
mint.narrow(self.cos_cached, 0, 0, seq_len).to(dtype=x.dtype)
mint.narrow(position_ids, 1, position_ids.shape[1] - input_ids.shape[1], input_ids.shape[1])
```

**收益**：mint.narrow在昇腾上有专门的算子实现，避免切片操作的额外内存拷贝。

---

### 2. FlashAttention集成优化

#### 2.1 自动检测并启用FlashAttention

```python
# 在半精度下自动启用FlashAttention
if query_states.dtype == mindspore.half and not output_attentions:
    attn_output = mindspore.ops.flash_attention_score(
        query_states, key_states, value_states,
        query_states.shape[1],
        input_layout='BNSD',
        scalar_value=1/math.sqrt(self.head_dim),
        attn_mask=mint.narrow(attention_mask, -1, 0, key_states.shape[-2]).bool()
    )
else:
    # 常规注意力计算路径
    ...
```

**收益**：
- Prefill阶段显著降低时延
- 减少注意力计算的内存占用
- 利用昇腾NPU的FlashAttention硬件加速单元

#### 2.2 FlashAttention2实现（DeepSeek模型）

为DeepSeek模型新增`DeepseekFlashAttention`类，支持：
- 自动回退机制（不支持时回退到常规注意力）
- 设备兼容性检测
- 与KV-Cache无缝集成

---

### 3. MoE路由与专家计算优化

#### 3.1 Qwen2-MoE: einsum批量专家计算

**核心优化思路**：将循环遍历专家改为批量矩阵运算

```python
# 优化前：逐专家循环计算
for expert_idx in range(self.num_experts):
    expert_layer = self.experts[expert_idx]
    idx, top_x = ops.nonzero(expert_mask[expert_idx], as_tuple=True)
    if 0 not in idx.shape:
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states)

# 优化后：使用einsum批量计算
# 预先堆叠专家权重
self.w1 = mint.stack([self.experts[i].gate_proj.weight.T for i in range(self.num_experts)])
self.w2 = mint.stack([self.experts[i].down_proj.weight.T for i in range(self.num_experts)])
self.w3 = mint.stack([self.experts[i].up_proj.weight.T for i in range(self.num_experts)])

# 批量计算所有专家输出
hidden_w1 = ops.einsum('th,ehi->tei', hidden_states, self.w1)
hidden_w3 = ops.einsum('th,ehi->tei', hidden_states, self.w3)
activated = self.act(hidden_w1) * hidden_w3
expert_outputs = ops.einsum('tei,eih->teh', activated, self.w2)

# 加权求和
weighted_expert_outputs = expert_outputs * router_scores.unsqueeze(-1)
hidden_states = mint.sum(weighted_expert_outputs, dim=1)
```

**收益**：
- 消除Python循环开销
- 充分利用NPU的并行计算能力
- 显著降低Prefill和Decode时延

#### 3.2 DeepSeek-MoE: 分场景优化策略

针对Prefill和Decode两种场景采用不同策略：

**Decode阶段（少量token）**：
```python
if token_count <= decode_threshold:
    # 使用排序+分段处理，减少scatter开销
    sorted_indices = mindspore.ops.argsort(flat_expert_ids)
    # 按专家分组处理
    for idx, expert_id in enumerate(active_ids_np):
        expert_out = self.experts[int(expert_id)](expert_input)
        collected_updates.append(expert_out)
    # 批量scatter_add
    zero_output = zero_output.scatter_add(0, expanded_token_indices, combined_updates)
```

**Prefill阶段（大量token）**：
```python
else:
    # 使用capacity机制，构建固定大小的专家输入
    expert_inputs_flat = mint.zeros((num_expert_slots, hidden_dim), dtype=accum_dtype)
    expert_inputs_flat = expert_inputs_flat.scatter_add(0, positions_expanded, routed_inputs)
    expert_inputs = expert_inputs_flat.reshape((num_experts, capacity_value, hidden_dim))
    
    # 批量计算所有专家
    for expert_id in range(num_experts):
        expert_out = self.experts[expert_id](expert_inputs[expert_id])
        expert_outputs.append(expert_out)
```

**收益**：
- Decode阶段避免不必要的padding开销
- Prefill阶段最大化并行度
- 动态调整capacity值，平衡内存与性能

---

### 4. 数值稳定性优化

#### 4.1 RMSNorm使用PyBoost加速

```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(self.weight.dtype)
    if not self.training and USE_PYBOOST:
        return F.rms_norm(hidden_states, self.weight, self.variance_epsilon).to(input_dtype)
    variance = mint.mean(hidden_states.pow(2), -1, keepdim=True)
    hidden_states = hidden_states * mint.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

**收益**：推理时使用PyBoost优化的rms_norm，减少计算开销。

#### 4.2 MoE路由softmax稳定性

```python
# 使用稳定的topk，确保专家顺序与baseline一致
topk_weight, topk_idx = mint.topk(scores, self.top_k, dim=-1)
topk_weight = topk_weight.to(scores.dtype)
topk_idx = topk_idx.astype(mindspore.int32)

# 更高效的归一化
denominator = mint.sum(topk_weight, dim=-1, keepdim=True) + 1e-20
topk_weight = mint.div(topk_weight, denominator)
```

---

### 5. 内存优化

#### 5.1 专家权重惰性堆叠与释放

```python
if not self.copied:
    # 堆叠专家权重用于批量计算
    self.w1 = nn.Parameter(mint.stack([...]), requires_grad=False)
    self.w2 = nn.Parameter(mint.stack([...]), requires_grad=False)
    self.w3 = nn.Parameter(mint.stack([...]), requires_grad=False)
    self.copied = True
    # 释放原始专家模块，减少内存占用
    del self.experts
    gc.collect()
    self.experts = None
```

**收益**：
- 减少推理时的内存占用
- 避免重复的堆叠操作
- 首次推理后自动优化内存布局

#### 5.2 位置编码缓存优化

```python
# 缓存position_ids，避免decode阶段重复计算
self.cached_position_ids = None

if inputs_embeds.shape[1] == 1 and past_key_values:
    position_ids = mint.narrow(self.cached_position_ids, 1, -1, 1) + 1
```

#### 5.3 单token decode的causal_mask简化

```python
if inputs_embeds.shape[1] > 1:
    causal_mask = self._update_causal_mask(...)
else:
    # decode阶段简化mask计算
    causal_mask = (attention_mask[:, None, None, :] - 1) * float(ops.finfo(inputs_embeds.dtype).min)
```

**收益**：Decode阶段避免复杂的mask构建，降低时延。

---

### 6. Warmup预热机制（DeepSeek）

```python
def _run_warmup(self):
    """模型加载后自动预热，确保JIT编译稳定"""
    warmup_len = min(128, max_pos)
    
    with no_grad():
        # Prefill预热
        dummy_ids = mint.ones((1, warmup_len), dtype=mindspore.int32) * self.config.bos_token_id
        outputs = self.model(input_ids=dummy_ids, use_cache=True, ...)
        
        # Decode预热
        if cached_past is not None:
            decode_ids = mint.ones((1, 1), dtype=mindspore.int32) * self.config.bos_token_id
            self.model(input_ids=decode_ids, past_key_values=cached_past, ...)
    
    # 根据warmup结果设置MoE capacity
    self._set_moe_capacity_for_layers(moe_capacity)
```

**收益**：
- 避免首次推理的JIT编译延迟
- 动态确定最优的MoE capacity值
- 支持通过环境变量跳过（`DEEPSEEK_SKIP_WARMUP=1`）

---

## 📈 优化效果分析

### Prefill阶段优化

| 优化项 | 技术手段 | 预估收益 |
|-------|---------|---------|
| FlashAttention | 硬件加速 | 40-60% |
| einsum批量专家计算 | 消除循环 | 30-50% |
| mint算子替换 | 底层优化 | 10-20% |

### Decode阶段优化

| 优化项 | 技术手段 | 预估收益 |
|-------|---------|---------|
| 位置编码缓存 | 避免重复计算 | 15-25% |
| 简化causal_mask | 减少计算量 | 10-15% |
| 分场景MoE策略 | 减少padding | 20-30% |

### 内存优化

| 优化项 | 技术手段 | 预估收益 |
|-------|---------|---------|
| 专家权重堆叠复用 | 减少内存分配 | 10-20% |
| 原始专家模块释放 | gc.collect | 5-10% |

---

## 🔑 关键技术总结

1. **算子层优化**：全面使用mint算子，适配昇腾NPU特性
2. **注意力优化**：集成FlashAttention，显著提升长序列处理效率
3. **MoE优化**：einsum批量计算 + 分场景策略，充分利用并行能力
4. **内存优化**：惰性初始化 + 资源释放，降低显存占用
5. **数值稳定性**：PyBoost加速 + 稳定的softmax/topk实现

---


