import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 图像加载及预处理
def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size:
        image = image.resize((size, size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


# 反标准化函数，用于显示
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return tensor * std + mean


# 加载内容及风格图像
content_img = load_image("content5.jpg", size=512)
style_img = load_image("style.jpg", size=512)

# 使用预训练的VGG19，只保留特征提取部分，并冻结参数
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# 定义内容层和风格层对应的索引
content_layers = ['conv4_2']  # 对应索引21 (conv4_2)
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
# 手动映射层名到索引
layer_names = {
    '0': 'conv1_1', '2': 'conv1_2', '5': 'conv2_1', '7': 'conv2_2',
    '10': 'conv3_1', '12': 'conv3_2', '14': 'conv3_3', '16': 'conv3_4',
    '19': 'conv4_1', '21': 'conv4_2', '23': 'conv4_3', '25': 'conv4_4',
    '28': 'conv5_1', '30': 'conv5_2', '32': 'conv5_3', '34': 'conv5_4'
}
# 反向映射：层名 -> 索引
name_to_idx = {v: k for k, v in layer_names.items()}


# Gram矩阵计算
def gram_matrix(tensor):
    B, C, H, W = tensor.size()
    tensor = tensor.view(C, H * W)
    return torch.mm(tensor, tensor.t())


# 提取指定层的特征
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layer_names and layer_names[name] in layers:
            features[layer_names[name]] = x
    return features


# 计算损失
def calculate_loss(generated, content_features, style_features, content_weight=1, style_weight=1e6):
    gen_features = get_features(generated, vgg, content_layers + style_layers)

    content_loss = torch.mean((gen_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_layers:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.mean((gen_gram - style_gram) ** 2)

    return content_weight * content_loss + style_weight * style_loss


# 初始化生成图像（从内容图像复制）
generated_img = content_img.clone().requires_grad_(True)

# 提取内容和风格特征
content_features = get_features(content_img, vgg, content_layers + style_layers)
style_features = get_features(style_img, vgg, content_layers + style_layers)

# 优化器
optimizer = optim.Adam([generated_img], lr=0.1)

# 训练迭代
for step in range(500):
    optimizer.zero_grad()
    loss = calculate_loss(generated_img, content_features, style_features)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        generated_img.clamp_(-1.5, 1.5)

    if step % 50 == 0:
        print(f"Step [{step}/500] loss: {loss.item():.4f}")

# 显示最终图像
with torch.no_grad():
    denorm_img = denormalize(generated_img).squeeze().cpu().permute(1, 2, 0).clamp(0, 1)
plt.imshow(denorm_img)
plt.axis('off')
plt.show()