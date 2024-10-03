import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def initialize_weights_hu(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(output_dim, output_dim)
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.linear1(x))
        shortcut = self.shortcut(x)
        out = self.dropout(out)
        out = self.linear2(out)
        out += shortcut
        out = F.relu(out)
        return out

class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=None):
        if not hidden_dim:
            hidden_dim = input_size * 4
        super(CustomClassifier, self).__init__()
        self.block1 = ResidualBlock(input_size, hidden_dim)
        self.block2 = ResidualBlock(hidden_dim, hidden_dim)
        # self.intermediary = nn.Linear(hidden_dim, 256)  # Intermediary layer
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, embedding = None, mask=None):
        if embedding is not None:
            x = torch.cat((x, embedding), dim=1) # todo remove
        

        out = self.block1(x)
        out = self.dropout(out)  # Apply dropout after the second block
        out = self.block2(out)
        # out = F.relu(self.intermediary(out))  # Apply ReLU activation function after intermediary layer
        out = self.fc(out)
        return out

class CustomClassifierEmbLN(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, hidden_dim=None, dropout=0.10):
        if not hidden_dim:
            hidden_dim = (input_size + embedding_dim) * 2
        super(CustomClassifierEmbLN, self).__init__()
        self.block1 = ResidualBlock(input_size + embedding_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_size)  # Batch Normalization after embedding concatenation
        self.block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization after block2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, embedding, mask):
        device = x.device  # Get the device of x
        embedding = embedding.to(device)  # Move embedding to the same device as x
        x = self.bn1(x)  # Apply Batch Normalization after embedding concatenation
        x = torch.cat((x, embedding), dim=1)  # Concatenate embedding
        out = self.block1(x)
        out = self.ln1(out)  # Apply Layer Normalization after block1
        out = self.dropout(out)
        out = self.block2(out)
        out = self.bn2(out)  # Apply Batch Normalization after block2
        out = self.fc(out)
        return out

class CustomClassifierEmb2(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, hidden_dim=None, dropout=0.10):
        if not hidden_dim:
            hidden_dim = (input_size + embedding_dim) * 4
        super(CustomClassifierEmb2, self).__init__()
        self.block1 = ResidualBlock(input_size + embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_size)  # Batch Normalization after embedding concatenation
        self.block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization after block2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, embedding, mask):
        device = x.device  # Get the device of x
        embedding = embedding.to(device)  # Move embedding to the same device as x
        x = self.bn1(x)  # Apply Batch Normalization after embedding concatenation
        x = torch.cat((x, embedding), dim=1)  # Concatenate embedding
        out = self.block1(x)
        out = self.dropout(out)
        out = self.block2(out)
        out = self.bn2(out)  # Apply Batch Normalization after block2
        out = self.fc(out)
        return out

class CustomClassifierEmb(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, hidden_dim=None, dropout=0.10):
        if not hidden_dim:
            hidden_dim = (input_size + embedding_dim) * 2
        super(CustomClassifierEmb, self).__init__()
        self.block1 = ResidualBlock(input_size + embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_size)  # Batch Normalization after embedding concatenation
        self.block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization after block2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, embedding, mask):
        device = x.device  # Get the device of x
        embedding = embedding.to(device)  # Move embedding to the same device as x
        x = self.bn1(x)  # Apply Batch Normalization after embedding concatenation
        x = torch.cat((x, embedding), dim=1)  # Concatenate embedding
        out = self.block1(x)
        out = self.dropout(out)
        out = self.block2(out)
        out = self.bn2(out)  # Apply Batch Normalization after block2
        out = self.fc(out)
        return out


class CustomClassifierNorm(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=None, dropout=0.05):
        if not hidden_dim:
            hidden_dim = input_size * 4
        super(CustomClassifierNorm, self).__init__()
        self.block1 = ResidualBlock(input_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_size)  # Batch Normalization after block1
        self.block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization after block2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, embedding, mask):
        x = torch.cat((x, embedding), dim=1)  # todo remove
        x = self.bn1(x)
        out = self.block1(x)
        # out = self.bn1(out)  # Apply Batch Normalization after block1
        out = self.dropout(out)
        out = self.block2(out)
        # out = self.bn2(out)  # Apply Batch Normalization after block2
        out = self.fc(out)
        return out

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=0)
        return (x * weights).sum(dim=0)

class Net4(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, dropout_rate=0.1):
        super(Net4, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        input_size = input_size + embedding_dim
        self.attention = Attention(input_size)
        self.fc1 = nn.Linear(input_size, 512)  # Adjust for mask
        self.dropout1 = nn.Dropout(dropout_rate)   # Dropout layer after fc1
        self.fc2 = nn.Linear(512, 64)
        self.dropout2 = nn.Dropout(dropout_rate)   # Dropout layer after fc2
        self.fc3 = nn.Linear(64, output_size)
        # Residual connection for fc2
        self.shortcut = nn.Linear(input_size, 64)

    def forward(self, x, embedding, mask):
        device = x.device  # Get the device of x
        embedding = embedding.to(device)  # Move embedding to the same device as x
        x = self.batch_norm(x)
        x = x * mask  # Apply mask
        x = torch.cat((x, embedding), dim=1)  # todo remove
        # x = torch.cat((x, mask), dim=1)  # Concatenate input with mask
        identity = x  # Save the input for the residual connection
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        # Add the residual (identity) to the output of fc2 before activation
        identity = self.shortcut(identity)
        x = x + identity  # This is the residual connection
        x = self.fc3(x)
        return x

class Net2EmbNorm(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, dropout_rate=0.1):
        super(Net2EmbNorm, self).__init__()
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        self.batch_norm_embedding = nn.BatchNorm1d(embedding_dim)
        self.fc1 = nn.Linear(input_size + embedding_dim, 512)  # Adjust for mask and embedding
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after fc1
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after fc2
        self.fc3 = nn.Linear(256, output_size)
        # Residual connection for fc2
        self.shortcut = nn.Linear(input_size + embedding_dim, 256)  # Adjust for mask and embedding

    def forward(self, x, embedding, mask):
        x = self.batch_norm_input(x)
        embedding = self.batch_norm_embedding(embedding)
        x = x * mask  # Apply mask
        x = torch.cat((x, embedding), dim=1)  # Concatenate input with embedding
        identity = x  # Save the input for the residual connection
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        # Add the residual (identity) to the output of fc2 before activation
        identity = self.shortcut(identity)
        x = x + identity  # This is the residual connection
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, dropout_rate=0.1):
        super(Net2, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size + embedding_dim, 512)  # Adjust for mask and embedding
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after fc1
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after fc2
        self.fc3 = nn.Linear(256, output_size)
        # Residual connection for fc2
        self.shortcut = nn.Linear(input_size + embedding_dim, 256)  # Adjust for mask and embedding

    def forward(self, x, embedding, mask):
        device = x.device  # Get the device of x
        embedding = embedding.to(device)
        x = self.batch_norm(x)
        x = x * mask  # Apply mask
        x = torch.cat((x, embedding), dim=1)  # Concatenate input with embedding
        identity = x  # Save the input for the residual connection
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        # Add the residual (identity) to the output of fc2 before activation
        identity = self.shortcut(identity)
        x = x + identity  # This is the residual connection
        x = self.fc3(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, models, indices):
        if len(models)!=len(indices):
            raise ValueError('Must indicate index')
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.indices = indices

    def forward(self, x, embedding, mask):
        outputs = [model(x[:, indices], embedding, mask) for model, indices in zip(self.models, self.indices)]
        mean_output = torch.mean(torch.stack(outputs), dim=0)
        return mean_output



def get_perturbed_model(net, factor=0.15):
    def enable_dropout(model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    embedding_class = net.__class__.__name__.endswith('Emb')
    if not net.__class__.__name__.startswith('Net') and not embedding_class:
        return None
    import copy
    import torch
    import numpy as np

    # Clone the model

    perturbed_model = copy.deepcopy(net)

    if embedding_class:
        enable_dropout(perturbed_model)
        return perturbed_model

    # Identify the target layer (e.g., fc1)
    target_layer = perturbed_model.fc1
    # mean_weight = torch.std(target_layer.weight.data).item()
    noise_level = factor # Adjust the fraction as needed
    # Perturb the weights of the target layer
    target_layer.weight.data += torch.randn_like(target_layer.weight) * noise_level

    # Get model output
    return perturbed_model