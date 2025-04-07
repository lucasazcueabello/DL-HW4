from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        hidden_values = 25

        self.layer_stack = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=self.n_track*4, out_features=hidden_values),
            nn.ReLU(),
            nn.Linear(in_features=hidden_values, out_features=hidden_values),
            nn.ReLU(),
            nn.Linear(in_features=hidden_values, out_features=hidden_values),
            nn.ReLU(),
            nn.Linear(in_features=hidden_values, out_features=hidden_values),
            nn.ReLU(),
            nn.Linear(in_features=hidden_values, out_features=self.n_waypoints*2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        return self.layer_stack(torch.cat([track_left, track_right], dim=1)).reshape(-1,3,2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.input_embed = nn.Linear(in_features=2, out_features=d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.self_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=200, batch_first=True), num_layers=1)
        self.output_embed = nn.Linear(in_features=64, out_features=2)
        self.norm = nn.LayerNorm(d_model)


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        left = self.input_embed(track_left)
        right = self.input_embed(track_right)
        if torch.isnan(left).any():
            print("Nan element left")
            print(torch.isnan(left).any(dim=(1, 2)))
            print(left)
        if torch.isnan(right).any():
            print("Nan element right")
        x = torch.cat([left, right], dim=1)
        if torch.isnan(x).any():
            print("Nan element x")
        k = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        if torch.isnan(k).any():
            print("Nan in k")
        attn_output, _ = self.cross_attention(k, x, x)
        attn_output = self.norm(attn_output) 
        y = self.self_attention(attn_output, attn_output)
        #if torch.isnan(y2).any():
            #nan_positions = torch.nonzero(torch.isnan(y2), as_tuple=False)
            #for pos in nan_positions:
                #print(tuple(pos.tolist()))
                #print(y1[pos])
            #print("Nan element y2")
        #y = torch.nan_to_num(y, nan=0.5)
        
        return self.output_embed(y)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        in_channels: int = 3,
        num_blocks: int = 7
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        cnn_layers = [
            #torch.nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3),
            #torch.nn.GELU(),
        ]
        c1 = in_channels
        for _ in range(num_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, self.n_waypoints*2, kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)
        pass

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        logits = self.network(z)
        return logits.squeeze().reshape(logits.shape[0], self.n_waypoints,2)

    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.c3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.gelu = torch.nn.GELU()
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
            self.dropout = torch.nn.Dropout(p=0.1)

            if in_channels != out_channels or stride != 1:
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            else:
                self.residual = nn.Identity()

        def forward(self, x):
            x1 = self.dropout(self.gelu(self.batch_norm(self.c1(x))))
            x1 = self.dropout(self.gelu(self.batch_norm(self.c2(x1))))
            x1 = self.dropout(self.gelu(self.batch_norm(self.c3(x1))))
            x1 = self.dropout(self.gelu(self.batch_norm(self.c3(x1))))
            return x1 + self.residual(x)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
