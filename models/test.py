# import torch

# # 选择top K 的位置
# K = 5  # 选择前5个得分最高的位置
# x = torch.rand(3,3)
# topk_indices = torch.topk(x, k=K, dim=-1).indices  # 获取top K 的位置

# # 选择对应的attention_output中的值
# selected_output = torch.gather(attention_output, dim=2, index=topk_indices.unsqueeze(3).expand(-1, -1, -1, attention_output.size(-1)))

# # selected_output 现在包含了每个位置的 top K attention_output 值

import torch

# # 输入特征，shape: [4, 3, 2]
# window_attention = torch.tensor([[[1.0, 2.0], [0.5, 1.0], [2.0, 3.0]],
#                                  [[2.0, 1.0], [3.0, 1.5], [0.5, 2.0]],
#                                  [[0.5, 1.5], [1.5, 0.5], [3.0, 2.5]],
#                                  [[1.0, 0.5], [2.0, 1.0], [1.5, 2.0]]])

# # 选择每个窗口内最相关的两个位置的特征
# window_attention_topk, indices = window_attention.topk(2, dim=1)

# print("window_attention_topk shape:", window_attention_topk.shape)
# print("window_attention_topk:")
# print(window_attention_topk)

# print("indices shape:", indices.shape)
# print("indices:")
# print(indices)

x = torch.rand(1,3,3)
y = torch.rand(1,3,64)
print(x)
x = torch.softmax(x, dim=-1) * torch.softmax(x, dim=-2)
x = x.view(1,-1)
print(x)
topk,topk_indices = torch.topk(x, 2, dim=-1, largest=True)
print(topk)
print(topk_indices)
i_indices = topk_indices // 3
j_indices = topk_indices % 3
print(i_indices)
print(j_indices)