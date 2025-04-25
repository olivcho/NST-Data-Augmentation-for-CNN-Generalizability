import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import soundfile as sf

# Constants
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
MEL_FMIN = 0.0
MEL_FMAX = 8000.0
F0_MIN = 80
F0_MAX = 500

# Audio preprocessing utilities
class AudioProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, 
                 hop_length=HOP_LENGTH, n_mels=N_MELS):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            n_mels=n_mels
        )
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel-spectrogram from audio waveform."""
        if isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio)
        
        # Ensure audio is the right shape and dtype
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Convert to spectrogram and then to mel
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        
        return log_mel_spec.squeeze(0)  # Remove batch dimension
    
    def extract_pitch(self, audio):
        """Extract F0 (pitch) contour from audio waveform."""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        if audio.ndim > 1:
            audio = audio.squeeze()
            
        # Use PYIN algorithm for pitch extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=F0_MIN,
            fmax=F0_MAX,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Replace NaN values with 0
        f0 = np.nan_to_num(f0)
        
        return torch.FloatTensor(f0), torch.FloatTensor(voiced_flag)
    
    def load_audio(self, file_path):
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(file_path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        return waveform.squeeze(0)  # Remove channel dimension
    
    def normalize_audio(self, audio):
        """Normalize audio to have zero mean and unit variance."""
        return (audio - audio.mean()) / (audio.std() + 1e-8)
    
    def preprocess_audio(self, file_path):
        """Complete preprocessing pipeline for a single audio file."""
        # Load and normalize audio
        audio = self.load_audio(file_path)
        audio = self.normalize_audio(audio)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio)
        pitch, voiced = self.extract_pitch(audio)
        
        # Ensure pitch and mel-spec have the same length
        min_len = min(mel_spec.shape[1], len(pitch))
        mel_spec = mel_spec[:, :min_len]
        pitch = pitch[:min_len]
        voiced = voiced[:min_len]
        
        return {
            'audio': audio,
            'mel_spectrogram': mel_spec,
            'pitch': pitch,
            'voiced': voiced,
            'file_path': file_path
        }

# Dataset class
class AccentDataset(Dataset):
    def __init__(self, source_dir, target_dir, processor=None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.processor = processor if processor else AudioProcessor()
        
        # Get all matching audio files
        self.source_files = sorted(list(self.source_dir.glob('*.wav')))
        self.target_files = sorted(list(self.target_dir.glob('*.wav')))
        
        # For unpaired data, we'll randomly assign source and target
        if len(self.source_files) != len(self.target_files):
            print(f"Warning: Source ({len(self.source_files)}) and target "
                  f"({len(self.target_files)}) file counts don't match. "
                  "Using unpaired training approach.")
    
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        
        # For unpaired training, randomly select a target file
        target_idx = idx % len(self.target_files)
        target_file = self.target_files[target_idx]
        
        # Preprocess both files
        source_data = self.processor.preprocess_audio(source_file)
        target_data = self.processor.preprocess_audio(target_file)
        
        return {
            'source': source_data,
            'target': target_data
        }

# Model Architecture Components

class Encoder(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden_dim=256, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm3 = nn.InstanceNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, latent_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        # x shape: [batch_size, n_mels, time]
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        
        # Reshape for LSTM: [batch_size, time, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=N_MELS):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=5, padding=2)
        
    def forward(self, x):
        # x shape: [batch_size, time, input_dim]
        x, _ = self.lstm(x)
        
        # Reshape for Conv1d: [batch_size, hidden_dim*2, time]
        x = x.transpose(1, 2)
        
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        
        return x

class PitchPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(PitchPredictor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.InstanceNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, 2, kernel_size=5, padding=2)  # 2 outputs: pitch and voiced flag
        
    def forward(self, x):
        # x shape: [batch_size, time, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, time]
        
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        
        # Split outputs
        pitch = x[:, 0, :]  # [batch_size, time]
        voiced = torch.sigmoid(x[:, 1, :])  # [batch_size, time]
        
        return pitch, voiced

class Discriminator(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim//4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(hidden_dim//4, hidden_dim//2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    def forward(self, x):
        # x shape: [batch_size, n_mels, time]
        x = x.unsqueeze(1)  # [batch_size, 1, n_mels, time]
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x)
        
        return x

# Main Accent Conversion Model
class AccentConversionModel(nn.Module):
    def __init__(self):
        super(AccentConversionModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pitch_predictor = PitchPredictor()
        self.discriminator = Discriminator()
        
        # Initialize vocoder (WaveRNN or other neural vocoder would go here)
        # For simplicity, we'll use Griffin-Lim algorithm for now
        self.mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            n_mels=N_MELS, 
            fmin=MEL_FMIN, 
            fmax=MEL_FMAX
        )
        
    def forward(self, source_mel, target_mel=None, source_pitch=None):
        # Encode source mel-spectrogram
        encoded = self.encoder(source_mel)
        
        # Predict pitch modification
        if source_pitch is not None:
            predicted_pitch, predicted_voiced = self.pitch_predictor(encoded)
        else:
            predicted_pitch, predicted_voiced = None, None
        
        # Decode to target mel-spectrogram
        decoded_mel = self.decoder(encoded)
        
        # If in training mode, compute discriminator output
        if target_mel is not None:
            real_disc = self.discriminator(target_mel)
            fake_disc = self.discriminator(decoded_mel)
        else:
            real_disc, fake_disc = None, None
        
        return {
            'encoded': encoded,
            'decoded_mel': decoded_mel,
            'predicted_pitch': predicted_pitch,
            'predicted_voiced': predicted_voiced,
            'real_disc': real_disc,
            'fake_disc': fake_disc
        }
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel-spectrogram to audio using Griffin-Lim algorithm."""
        # Convert log-mel to linear magnitude spectrogram
        mel_spectrogram = mel_spectrogram.detach().cpu().numpy()
        mag_spectrogram = np.dot(librosa.util.pinv(self.mel_basis), np.power(10, mel_spectrogram))
        
        # Apply Griffin-Lim to recover phase
        audio = librosa.griffinlim(
            mag_spectrogram, 
            n_iter=50,
            hop_length=HOP_LENGTH,
            win_length=N_FFT
        )
        
        return audio

# Training code
def train_model(model, dataset, num_epochs=100, batch_size=4, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizers
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + 
        list(model.decoder.parameters()) + 
        list(model.pitch_predictor.parameters()), 
        lr=learning_rate
    )
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            source_mel = batch['source']['mel_spectrogram'].to(device)
            target_mel = batch['target']['mel_spectrogram'].to(device)
            source_pitch = batch['source']['pitch'].to(device)
            target_pitch = batch['target']['pitch'].to(device)
            source_voiced = batch['source']['voiced'].to(device)
            target_voiced = batch['target']['voiced'].to(device)
            
            # Format check and adjustments
            if source_mel.dim() == 2:
                source_mel = source_mel.unsqueeze(0)
            if target_mel.dim() == 2:
                target_mel = target_mel.unsqueeze(0)
            
            # Train discriminator
            optimizer_D.zero_grad()
            
            # Get model outputs
            outputs = model(source_mel, target_mel, source_pitch)
            
            # Real samples -> 1, Fake samples -> 0
            real_target = torch.ones_like(outputs['real_disc'])
            fake_target = torch.zeros_like(outputs['fake_disc'])
            
            d_real_loss = bce_loss(outputs['real_disc'], real_target)
            d_fake_loss = bce_loss(outputs['fake_disc'], fake_target)
            d_loss = d_real_loss + d_fake_loss
            
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            
            # Train generator
            optimizer_G.zero_grad()
            
            # Reconstruction loss for mel-spectrogram
            mel_loss = l1_loss(outputs['decoded_mel'], target_mel)
            
            # Pitch prediction loss
            pitch_loss = mse_loss(outputs['predicted_pitch'], target_pitch)
            voiced_loss = bce_loss(outputs['predicted_voiced'], target_voiced)
            
            # Adversarial loss (fool the discriminator)
            adv_loss = bce_loss(outputs['fake_disc'], real_target)
            
            # Total generator loss
            g_loss = mel_loss + 0.1 * pitch_loss + 0.1 * voiced_loss + 0.01 * adv_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            # Track losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                      f"G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Print epoch summary
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed - "
              f"Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_G.state_dict(),
                'optimizer_d_state_dict': optimizer_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, f'accent_conversion_model_epoch_{epoch+1}.pt')
            
            # Generate and save samples
            with torch.no_grad():
                sample_idx = np.random.randint(0, len(dataset))
                sample = dataset[sample_idx]
                source_mel = sample['source']['mel_spectrogram'].unsqueeze(0).to(device)
                
                outputs = model(source_mel)
                converted_mel = outputs['decoded_mel'][0].cpu()
                
                # Convert to audio and save
                source_audio = sample['source']['audio'].numpy()
                converted_audio = model.mel_to_audio(converted_mel)
                
                sf.write(f'sample_epoch_{epoch+1}_source.wav', source_audio, SAMPLE_RATE)
                sf.write(f'sample_epoch_{epoch+1}_converted.wav', converted_audio, SAMPLE_RATE)

# Inference function
def convert_accent(model, audio_file, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    processor = AudioProcessor()
    
    # Preprocess audio
    data = processor.preprocess_audio(audio_file)
    mel_spec = data['mel_spectrogram'].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(mel_spec)
        converted_mel = outputs['decoded_mel'][0].cpu()
    
    # Convert to audio
    converted_audio = model.mel_to_audio(converted_mel)
    
    # Save output
    sf.write(output_file, converted_audio, SAMPLE_RATE)
    
    return converted_audio

# Example usage
if __name__ == "__main__":
    # Create model
    model = AccentConversionModel()
    
    # Create dataset
    # You'll need to provide paths to directories containing neutral and Indian-accented speech
    dataset = AccentDataset(
        source_dir="path/to/neutral_accent",
        target_dir="path/to/indian_accent"
    )
    
    # Train model
    train_model(model, dataset, num_epochs=100, batch_size=4, learning_rate=0.0001)
    
    # Inference example
    convert_accent(model, "test_audio.wav", "test_audio_indian_accent.wav")