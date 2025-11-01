import torch
from torch import nn



class Embedding(nn.Module):
    def __init__(self, image_size:int=224, input_channels:int=3, patch_size:int=16, 
                 embedding_dropout:float=0.1):
        super().__init__()

        assert image_size%patch_size==0, f"Image size must be divisible by the Patch size.\nImage size:{image_size}, Patch size:{patch_size}"

        self.embedding_dim = int((patch_size**2)*input_channels)
        self.num_patches = int(((image_size//patch_size)**2))

        # Patch Embedder (input size: (bs, 3, 224, 224), output size: (bs, 768, 14, 14))
        self.patch_embedder = nn.Conv2d(in_channels=input_channels, out_channels=self.embedding_dim,
                                        kernel_size=patch_size, stride=patch_size, padding=0) # 📌📌📌 Padding Options
        # Flatten the patched image (input size: (bs, 768, 14, 14), output size: (bs, 768, 196))
        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # 📌📌📌 start_dim, end_dim
        # Class Embedder
        self.class_embedder = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim), requires_grad=True) # 📌📌📌 Why 1 in batch_size?
        # Position Embedder # 📌📌📌 Class and Position embedding shape is in this way. Why?
        self.position_embedder = nn.Parameter(data=torch.rand(1, self.num_patches+1, self.embedding_dim), requires_grad=True)
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout) 

    def forward(self, x:torch.tensor)->torch.tensor:
        # Patch Embedding
        x = self.patch_embedder(x); x = self.flatten(x); x = x.permute(0,2,1)
        # Class Embedding
        bs = x.shape[0]; class_token = self.class_embedder.expand(bs, -1, -1) # 📌📌📌 Didn't understand what's happening here.
        x = torch.cat((x, class_token), dim=1)
        # Position Embedding
        x = x + self.position_embedder
        # Embedding dropout
        x = self.embedding_dropout(x)
        return x
    


class EncoderBlock(nn.Module):
    def __init__(self, input_channels:int=3, patch_size:int=16, 
                 num_heads:int=12, mlp_size:int=3072, 
                 attention_dropout:float=0.1, mlp_dropout:float=0.1):
        super().__init__()

        self.embedding_dim = int((patch_size**2)*input_channels)

        # Normalisation for Attention Layer
        self.attention_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        # Normalistaion for MLP layer
        self.mlp_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        # Multi-head Self Attention Layer
        self.msa_layer = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, 
                                               dropout=attention_dropout, batch_first=True)
        # MLP layer
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=self.embedding_dim), # 📌📌📌 Why not GELU after it?
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x:torch.tensor) -> torch.tensor:
        # Normalisation before Attention Layer
        x_norm = self.attention_norm(x)
        # Attention Layer
        attention_output, _ = self.msa_layer(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        # Add attention output to the original x
        x = x + attention_output

        # Normalisation before MLP layer
        x_norm = self.mlp_norm(x)
        # MLP layer
        mlp_output = self.mlp_layer(x_norm)
        # Add to x + attention output
        x = x + mlp_output # 📌📌📌 Don't we need layer normalisation of the output before entering to the 
        return x        # 📌📌📌 next encoder(normalisation at begining of it) or classifier(should be normalised first before entering)?
    


class VisionTransformer(nn.Module):
    def __init__(self, image_size:int=224, input_channels:int=3, patch_size:int=16,
                 num_heads:int=12, mlp_size:int=3072, num_classes:int=1000, num_layers:int=12,
                 embedding_dropout:float=0.1, attention_dropout:float=0.1, mlp_dropout:float=0.1):
        super().__init__()

        self.embedding_dim = int((patch_size**2)*input_channels)
        
        # Patch + Class + Position Embedding Block
        self.embedding_block = Embedding(image_size=image_size, input_channels=input_channels, patch_size=patch_size,
                                         embedding_dropout=embedding_dropout)
        # Encoder Block
        self.encoder_block = nn.Sequential(*[EncoderBlock(input_channels=input_channels, patch_size=patch_size, 
                                          num_heads=num_heads, mlp_size=mlp_size, 
                                          attention_dropout=attention_dropout, mlp_dropout=mlp_dropout) for _ in range(num_layers)])
        # Classifier Block
        self.classifier_block = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=num_classes)
        )

    def forward(self, x:torch.tensor)->torch.tensor:
        # Embedding Block
        x = self.embedding_block(x)
        # Encoder Block
        x = self.encoder_block(x)
        # Classifier Block
        x = self.classifier_block(x[:,0]) # 📌📌📌 Why x[:,0]?
        return x