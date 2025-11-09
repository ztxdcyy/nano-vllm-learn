import math
import torch

def safe_softmax(z):
    """
    纯 Python 实现的 SafeSoftmax 函数
    :param z: 输入列表或元组（如 [1, 3, 2, 4]）
    :return: 归一化后的概率列表
    """
    # 步骤1：找到输入向量中的最大值（用于数值稳定）
    max_val = max(z)
    
    # 步骤2：计算每个元素减去最大值后的指数（避免溢出）
    exp_shifted = [math.exp(x - max_val) for x in z]
    
    # 步骤3：计算分母（所有指数的和）
    denominator = sum(exp_shifted)
    
    # 步骤4：归一化得到概率
    return [x / denominator for x in exp_shifted]

def safe_softmax_pytorch(z:torch.Tensor, dim:int=-1):
    """
    手动实现 SafeSoftmax，避免数值溢出
    :param z: 输入张量（可以是批量的，如 [batch_size, seq_len, hidden_dim]）
    :param dim: 计算 Softmax 的维度
    :return: 归一化后的概率张量
    """
    # 步骤1：计算输入在指定维度上的最大值（用于平移）
    max_val = torch.max(z, dim=dim, keepdim=True).values  # keepdim=True 保持维度一致，方便广播
    
    # 步骤2：输入减去最大值（平移，避免指数溢出）
    z_shifted = z - max_val
    
    # 步骤3：计算指数和分母（dN）
    exp_z = torch.exp(z_shifted)
    dN = torch.sum(exp_z, dim=dim, keepdim=True)  # 分母：所有指数的和
    
    # 步骤4：归一化得到概率
    return exp_z / dN

# 测试示例
if __name__ == "__main__":
    # 输入向量（可以是任意实数列表）
    z = [1, 3, 2, 4]
    
    # 计算 SafeSoftmax
    # result = safe_softmax(z)
    z = torch.tensor(z)
    result = safe_softmax_pytorch(z)
    
    print("输入向量：", z)
    print("SafeSoftmax 结果：", [round(x.item(), 4) for x in result])  # 保留4位小数
    print("概率和：", round(sum(result).item(), 6))  # 验证和为1（浮点数精度范围内）