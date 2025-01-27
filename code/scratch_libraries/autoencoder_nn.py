from torch import nn

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()  # Output values in range [0, 1]
        )

    def encode(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Encode
        encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        # Decode
        decoded = self.decoder(encoded)
        # Reshape to original image dimensions
        decoded = decoded.view(decoded.size(0), 1, 28, 28)
        return decoded

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded




