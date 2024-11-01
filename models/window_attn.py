import torch
import torch.nn.functional as F

class WindowAttentionLayer(torch.nn.Module):
    def __init__(self, window_size, input_dim, num_heads):
        super(WindowAttentionLayer, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Learnable parameters for Q, K, and V
        self.W_q = torch.nn.Linear(input_dim, input_dim)
        self.W_k = torch.nn.Linear(input_dim, input_dim)
        self.W_v = torch.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Split the input into heads
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Calculate attention scores using dot product
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)

        # Define a window mask (adjust this for your specific task)
        window_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=self.window_size)
        window_mask = window_mask.to(x.device)

        # Apply the window mask to the attention scores
        scores = scores.masked_fill(window_mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to the values
        output = torch.matmul(attention_weights, v)

        return output

def top_k_patches(attention_output, k):
    # Flatten the attention output
    flattened = attention_output.view(attention_output.size(0), -1, attention_output.size(-1))
    
    # Get the indices of the top-k values
    top_k_values, top_k_indices = torch.topk(flattened, k, dim=1)
    
    # Reshape the indices to match the original shape
    """ 这里选择索引和值"""
    batch_indices = top_k_indices // flattened.size(-1)  
    position_indices = top_k_indices % flattened.size(-1)
    
    return batch_indices, position_indices

# Example usage
input_dim = 64
seq_len = 16
num_heads = 4
window_size = 4
k = 3

layer = WindowAttentionLayer(window_size, input_dim, num_heads)
input_data = torch.randn(32, seq_len, input_dim)
output = layer(input_data)
top_k_batch_indices, top_k_position_indices = top_k_patches(output, k)
