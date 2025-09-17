
#!/usr/bin/env python3
"""
音频处理程序 - Audio Processing Program
一个功能全面的音频处理工具，支持多种音频操作
A comprehensive audio processing tool with various audio operations
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """音频处理器类"""
    
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.duration = None
        self.channels = None
        
    def load_audio(self, file_path):
        """加载音频文件"""
        try:
            print(f"正在加载音频文件: {file_path}")
            
            # 使用librosa加载音频文件
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # 处理单声道/立体声
            if len(self.audio_data.shape) == 1:
                self.audio_data = self.audio_data.reshape(1, -1)
                self.channels = 1
            else:
                self.channels = self.audio_data.shape[0]
            
            self.duration = self.audio_data.shape[1] / self.sample_rate
            
            print(f"音频文件加载成功!")
            print(f"采样率: {self.sample_rate} Hz")
            print(f"声道数: {self.channels}")
            print(f"时长: {self.duration:.2f} 秒")
            print(f"数据形状: {self.audio_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"加载音频文件失败: {e}")
            return False
    
    def save_audio(self, output_path, format='wav'):
        """保存音频文件"""
        try:
            if self.audio_data is None:
                print("没有音频数据可保存")
                return False
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 如果是单声道，去掉多余的维度
            if self.channels == 1:
                audio_to_save = self.audio_data[0]
            else:
                audio_to_save = self.audio_data.T
            
            sf.write(output_path, audio_to_save, self.sample_rate, format=format)
            print(f"音频文件已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存音频文件失败: {e}")
            return False
    
    def get_audio_info(self):
        """获取音频信息"""
        if self.audio_data is None:
            print("没有加载音频文件")
            return None
        
        info = {
            '采样率': self.sample_rate,
            '声道数': self.channels,
            '时长': f"{self.duration:.2f} 秒",
            '样本数': self.audio_data.shape[1],
            '数据类型': str(self.audio_data.dtype),
            '最大值': np.max(np.abs(self.audio_data)),
            'RMS': np.sqrt(np.mean(self.audio_data**2))
        }
        
        print("\n=== 音频信息 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        return info
    
    def normalize_audio(self, target_level=0.8):
        """音频归一化"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return False
            
            print("正在执行音频归一化...")
            
            # 对每个声道分别归一化
            for channel in range(self.channels):
                channel_data = self.audio_data[channel]
                max_val = np.max(np.abs(channel_data))
                if max_val > 0:
                    self.audio_data[channel] = channel_data / max_val * target_level
            
            print("音频归一化完成")
            return True
            
        except Exception as e:
            print(f"音频归一化失败: {e}")
            return False
    
    def apply_filter(self, filter_type, cutoff_freq, order=5):
        """应用滤波器"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return False
            
            print(f"正在应用{filter_type}滤波器，截止频率: {cutoff_freq} Hz")
            
            nyquist = self.sample_rate / 2
            normal_cutoff = cutoff_freq / nyquist
            
            # 设计滤波器
            if filter_type.lower() == 'low':
                b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            elif filter_type.lower() == 'high':
                b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            elif filter_type.lower() == 'band':
                b, a = signal.butter(order, normal_cutoff, btype='band', analog=False)
            else:
                print(f"不支持的滤波器类型: {filter_type}")
                return False
            
            # 应用滤波器
            for channel in range(self.channels):
                self.audio_data[channel] = signal.filtfilt(b, a, self.audio_data[channel])
            
            print(f"{filter_type}滤波器应用完成")
            return True
            
        except Exception as e:
            print(f"滤波器应用失败: {e}")
            return False
    
    def change_volume(self, volume_factor):
        """改变音量"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return False
            
            print(f"正在调整音量，倍数: {volume_factor}")
            
            self.audio_data = self.audio_data * volume_factor
            
            # 防止削波
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 1.0:
                print("警告: 音量过大，发生削波，正在自动调整...")
                self.audio_data = self.audio_data / max_val * 0.98
            
            print("音量调整完成")
            return True
            
        except Exception as e:
            print(f"音量调整失败: {e}")
            return False
    
    def extract_features(self):
        """提取音频特征"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return None
            
            print("正在提取音频特征...")
            
            # 使用第一个声道进行特征提取
            if self.channels > 1:
                audio_mono = np.mean(self.audio_data, axis=0)
            else:
                audio_mono = self.audio_data[0]
            
            features = {}
            
            # 基本统计特征
            features['mean'] = np.mean(audio_mono)
            features['std'] = np.std(audio_mono)
            features['max'] = np.max(audio_mono)
            features['min'] = np.min(audio_mono)
            
            # 零交叉率
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_mono))
            
            # 频谱特征
            stft = librosa.stft(audio_mono)
            magnitude = np.abs(stft)
            
            # 光谱质心
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate))
            
            # 光谱带宽
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate))
            
            # 光谱滚降
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate))
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio_mono, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=audio_mono, sr=self.sample_rate)
            features['chroma_mean'] = np.mean(chroma)
            
            # 音调特征
            features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=audio_mono, sr=self.sample_rate))
            
            print("\n=== 音频特征 ===")
            for key, value in features.items():
                print(f"{key}: {value:.6f}")
            
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def plot_waveform(self, save_path=None):
        """绘制波形图"""
        try:
            if self.audio_data is None:
                print("没有音频数据可绘制")
                return False
            
            print("正在绘制波形图...")
            
            plt.figure(figsize=(12, 8))
            
            # 绘制波形
            for channel in range(self.channels):
                plt.subplot(self.channels, 1, channel + 1)
                time_axis = np.linspace(0, self.duration, len(self.audio_data[channel]))
                plt.plot(time_axis, self.audio_data[channel])
                plt.title(f'声道 {channel + 1} 波形')
                plt.xlabel('时间 (秒)')
                plt.ylabel('振幅')
                plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"波形图已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"绘制波形图失败: {e}")
            return False
    
    def plot_spectrum(self, save_path=None):
        """绘制频谱图"""
        try:
            if self.audio_data is None:
                print("没有音频数据可绘制")
                return False
            
            print("正在绘制频谱图...")
            
            plt.figure(figsize=(12, 8))
            
            for channel in range(self.channels):
                plt.subplot(self.channels, 1, channel + 1)
                
                # 计算FFT
                fft_data = np.fft.fft(self.audio_data[channel])
                freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
                
                # 只显示正频率部分
                positive_freqs = freqs[:len(freqs)//2]
                magnitude = np.abs(fft_data[:len(fft_data)//2])
                
                plt.plot(positive_freqs, magnitude)
                plt.title(f'声道 {channel + 1} 频谱')
                plt.xlabel('频率 (Hz)')
                plt.ylabel('幅度')
                plt.xlim(0, self.sample_rate//2)
                plt.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"频谱图已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"绘制频谱图失败: {e}")
            return False
    
    def plot_spectrogram(self, save_path=None):
        """绘制语谱图"""
        try:
            if self.audio_data is None:
                print("没有音频数据可绘制")
                return False
            
            print("正在绘制语谱图...")
            
            plt.figure(figsize=(12, 8))
            
            for channel in range(self.channels):
                plt.subplot(self.channels, 1, channel + 1)
                
                # 计算语谱图
                D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data[channel])), 
                                          ref=np.max)
                
                librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'声道 {channel + 1} 语谱图')
                plt.xlabel('时间')
                plt.ylabel('频率 (Hz)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"语谱图已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"绘制语谱图失败: {e}")
            return False
    
    def trim_silence(self, threshold=0.01):
        """去除静音部分"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return False
            
            print("正在去除静音部分...")
            
            for channel in range(self.channels):
                # 找到非静音部分的索引
                non_silent = np.abs(self.audio_data[channel]) > threshold
                indices = np.where(non_silent)[0]
                
                if len(indices) > 0:
                    start_idx = indices[0]
                    end_idx = indices[-1]
                    self.audio_data[channel] = self.audio_data[channel][start_idx:end_idx+1]
                else:
                    print(f"警告: 声道 {channel + 1} 检测到全是静音")
            
            # 更新时长
            self.duration = self.audio_data.shape[1] / self.sample_rate
            
            print("静音部分去除完成")
            print(f"处理后时长: {self.duration:.2f} 秒")
            return True
            
        except Exception as e:
            print(f"去除静音失败: {e}")
            return False
    
    def fade_in_out(self, fade_in_duration=0.1, fade_out_duration=0.1):
        """添加淡入淡出效果"""
        try:
            if self.audio_data is None:
                print("没有音频数据可处理")
                return False
            
            print(f"正在添加淡入淡出效果: 淡入 {fade_in_duration}秒, 淡出 {fade_out_duration}秒")
            
            fade_in_samples = int(fade_in_duration * self.sample_rate)
            fade_out_samples = int(fade_out_duration * self.sample_rate)
            
            for channel in range(self.channels):
                # 淡入
                if fade_in_samples > 0:
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                    self.audio_data[channel][:fade_in_samples] *= fade_in_curve
                
                # 淡出
                if fade_out_samples > 0:
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                    self.audio_data[channel][-fade_out_samples:] *= fade_out_curve
            
            print("淡入淡出效果添加完成")
            return True
            
        except Exception as e:
            print(f"添加淡入淡出效果失败: {e}")
            return False


def main():
    """主函数 - 演示音频处理功能"""
    print("=== 音频处理程序 ===")
    print("Audio Processing Program")
    print("=" * 50)
    
    # 创建音频处理器实例
    processor = AudioProcessor()
    
    # 演示功能
    print("\n1. 创建示例音频文件...")
    # 创建一个简单的示例音频（正弦波）
    duration = 3.0  # 3秒
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建一个包含多个频率的复合音频
    frequency1 = 440  # A4音符
    frequency2 = 880  # A5音符
    audio_data = 0.5 * np.sin(2 * np.pi * frequency1 * t) + \
                 0.3 * np.sin(2 * np.pi * frequency2 * t)
    
    # 添加一些噪声
    noise = 0.05 * np.random.normal(0, 1, len(t))
    audio_data += noise
    
    # 保存示例音频
    example_file = "/workspace/example_audio.wav"
    sf.write(example_file, audio_data, sample_rate)
    print(f"示例音频文件已创建: {example_file}")
    
    # 加载音频文件
    print("\n2. 加载音频文件...")
    if processor.load_audio(example_file):
        # 获取音频信息
        print("\n3. 获取音频信息...")
        processor.get_audio_info()
        
        # 绘制波形图
        print("\n4. 绘制波形图...")
        processor.plot_waveform("/workspace/waveform.png")
        
        # 绘制频谱图
        print("\n5. 绘制频谱图...")
        processor.plot_spectrum("/workspace/spectrum.png")
        
        # 绘制语谱图
        print("\n6. 绘制语谱图...")
        processor.plot_spectrogram("/workspace/spectrogram.png")
        
        # 提取音频特征
        print("\n7. 提取音频特征...")
        features = processor.extract_features()
        
        # 音频归一化
        print("\n8. 音频归一化...")
        processor.normalize_audio()
        
        # 应用低通滤波器
        print("\n9. 应用低通滤波器...")
        processor.apply_filter('low', 1000)
        
        # 添加淡入淡出效果
        print("\n10. 添加淡入淡出效果...")
        processor.fade_in_out(0.2, 0.2)
        
        # 保存处理后的音频
        print("\n11. 保存处理后的音频...")
        processor.save_audio("/workspace/processed_audio.wav")
        
        print("\n=== 处理完成 ===")
        print("生成的文件:")
        print("- example_audio.wav: 原始示例音频")
        print("- waveform.png: 波形图")
        print("- spectrum.png: 频谱图")
        print("- spectrogram.png: 语谱图")
        print("- processed_audio.wav: 处理后的音频")
    
    else:
        print("音频文件加载失败")


if __name__ == "__main__":
    main()
