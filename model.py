# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    """
    Encodes the entire time series into a single latent vector using an LSTM.
    Suitable for capturing static, time-invariant features.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(TemporalEncoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim, seq_len] -> [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.rnn(x)
        # Concatenate the final hidden states of the last layer from both directions
        h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

class TemporalDecoder(nn.Module):
    """
    Decodes a latent representation back into a time series.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super(TemporalDecoder, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        
    def forward(self, z):
        # z is the aggregated latent representation
        h = self.fc(z).view(z.size(0), -1, self.seq_len)
        h = F.relu(self.deconv1(h))
        return self.deconv2(h)

class CoTDVAE(nn.Module):
    """
    Contrastive Temporal Disentanglement Variational Autoencoder (CoTD-VAE).

    This model disentangles a time series into three components:
    1. Static: Time-invariant global properties.
    2. Trend: Smoothly varying temporal dynamics.
    3. Event: Sparse, sudden changes or anomalies.
    """
    def __init__(
        self, input_dim, seq_len, hidden_dim=64, latent_static_dim=16, latent_trend_dim=16, latent_event_dim=16,
        beta_static=1.0, beta_trend=1.0, beta_event=1.0,
        lambda_smooth=1.0, lambda_sparse=1.0,
        alpha_smooth=0.5, gamma_contrast=0.2, gamma_peak=0.2
    ):
        super(CoTDVAE, self).__init__()
        self.input_dim, self.seq_len = input_dim, seq_len
        
        # Hyperparameters for loss components
        self.beta_static, self.beta_trend, self.beta_event = beta_static, beta_trend, beta_event
        self.lambda_smooth, self.lambda_sparse = lambda_smooth, lambda_sparse
        self.alpha_smooth, self.gamma_contrast, self.gamma_peak = alpha_smooth, gamma_contrast, gamma_peak
        
        # Encoders for each component
        self.static_encoder = TemporalEncoder(input_dim, hidden_dim, latent_static_dim)
        self.trend_encoder = self._build_series_encoder(input_dim, hidden_dim, latent_trend_dim)
        self.event_encoder = self._build_series_encoder(input_dim, hidden_dim, latent_event_dim)
        
        # Decoder
        total_latent_dim = latent_static_dim + latent_trend_dim + latent_event_dim
        self.decoder = TemporalDecoder(total_latent_dim, hidden_dim, input_dim, seq_len)

    def _build_series_encoder(self, input_dim, hidden_dim, latent_dim):
        """Builds a 1D CNN encoder for time-varying components (trend and event)."""
        return nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim * 2, kernel_size=5, padding=2) # Outputs mu and logvar concatenated
        )

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode into static, trend, and event components
        static_mu, static_logvar = self.static_encoder(x)
        trend_params = self.trend_encoder(x)
        event_params = self.event_encoder(x)
        
        # Split the output of series encoders into mu and logvar
        trend_mu, trend_logvar = trend_params.chunk(2, dim=1)
        event_mu, event_logvar = event_params.chunk(2, dim=1)
        
        # 2. Sample from latent distributions
        z_static = self.reparameterize(static_mu, static_logvar)
        z_trend = self.reparameterize(trend_mu, trend_logvar)
        z_event = self.reparameterize(event_mu, event_logvar)
        
        # 3. Combine latent variables and decode
        # Expand static latent to match the sequence length for combination
        z_static_expanded = z_static.unsqueeze(2).expand(-1, -1, self.seq_len)
        z_combined_series = torch.cat([z_static_expanded, z_trend, z_event], dim=1)
        
        # The decoder takes a single vector, so we average across the time dimension
        z_combined_aggregated = torch.mean(z_combined_series, dim=2)
        x_recon = self.decoder(z_combined_aggregated)
        
        # 4. Calculate losses
        recon_loss = F.mse_loss(x_recon, x)
        kl_static = -0.5 * torch.mean(torch.sum(1 + static_logvar - static_mu.pow(2) - static_logvar.exp(), dim=1))
        kl_trend = -0.5 * torch.mean(torch.sum(1 + trend_logvar - trend_mu.pow(2) - trend_logvar.exp(), dim=[1,2]))
        kl_event = -0.5 * torch.mean(torch.sum(1 + event_logvar - event_mu.pow(2) - event_logvar.exp(), dim=[1,2]))
        
        L_smooth = self.compute_trend_smoothness(z_trend)
        L_sparse = self.compute_event_sparsity(z_event)
        
        total_loss = (recon_loss + 
                      self.beta_static * kl_static + 
                      self.beta_trend * kl_trend + 
                      self.beta_event * kl_event +
                      self.lambda_smooth * L_smooth + 
                      self.lambda_sparse * L_sparse)
                      
        loss_dict = {
            'total_loss': total_loss, 'recon_loss': recon_loss, 'kl_static': kl_static, 
            'kl_trend': kl_trend, 'kl_event': kl_event, 'L_smooth': L_smooth, 'L_sparse': L_sparse
        }
        
        return loss_dict, z_static, z_trend, z_event

    def compute_trend_smoothness(self, z_trend):
        """Computes smoothness loss for the trend component by penalizing derivatives."""
        # Penalize first-order difference (velocity)
        diff1 = z_trend[:, :, 1:] - z_trend[:, :, :-1]
        loss1 = torch.mean(diff1.pow(2))
        
        # Penalize second-order difference (acceleration)
        loss2 = 0.0
        if z_trend.size(2) > 2:
            diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
            loss2 = torch.mean(diff2.pow(2))
            
        return loss1 + self.alpha_smooth * loss2

    def compute_event_sparsity(self, z_event):
        """Computes sparsity loss for the event component."""
        # L1 norm to encourage sparsity
        l1_loss = torch.mean(torch.abs(z_event))
        
        # Contrastive loss to encourage high variance (some high, some low values)
        contrast_loss = -torch.mean(torch.std(z_event, dim=1))
        
        # Peakiness loss to encourage sharp peaks over the average
        max_val = torch.max(torch.abs(z_event), dim=2, keepdim=True)[0]
        avg_val = torch.mean(torch.abs(z_event), dim=2, keepdim=True)
        peak_loss = -torch.mean(max_val / (avg_val + 1e-8))
        
        return l1_loss + self.gamma_contrast * contrast_loss + self.gamma_peak * peak_loss

    @torch.no_grad()
    def extract_features(self, x):
        """
        Extracts the disentangled feature vectors for downstream tasks.
        Returns a single concatenated vector per time series.
        """
        self.eval()
        _, z_static, z_trend, z_event = self.forward(x)
        
        # Aggregate time-varying components to get a single vector
        z_trend_pooled = torch.mean(z_trend, dim=2)
        z_event_pooled = torch.mean(z_event, dim=2)
        
        # Concatenate all features
        return torch.cat([z_static, z_trend_pooled, z_event_pooled], dim=1)