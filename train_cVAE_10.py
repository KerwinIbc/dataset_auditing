import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, z_dim=128, num_classes=10):
        super().__init__()

        self.z_dim = z_dim
        self.num_classes = num_classes

        # ----- encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8->4
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 4 * 4 + num_classes, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4 + num_classes, z_dim)

        # ----- decoder -----
        self.fc_decode = nn.Linear(z_dim + num_classes, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 16->32
            nn.Tanh()
        )

    def encode(self, x, labels):
        B = x.size(0)
        h = self.encoder(x)
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        h = torch.cat([h, labels_onehot], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, labels):
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        z = torch.cat([z, labels_onehot], dim=1)

        h = self.fc_decode(z)
        h = h.view(z.size(0), 256, 4, 4)
        x = self.decoder(h)
        return x

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    # 给 inversion 用的接口
    def generate_from_z(self, z, labels):
        return self.decode(z, labels)


def train_cVAE(dataset, z_dim=128, num_classes=10, device='cuda', epochs=100, batch_size=128):

    cvae = ConditionalVAE(z_dim=z_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(cvae.parameters(), lr=1e-3)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        total_loss = 0

        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            x_recon, mu, logvar = cvae(x, y)

            # reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            loss = recon_loss + kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"[cVAE] Epoch {ep+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    return cvae
