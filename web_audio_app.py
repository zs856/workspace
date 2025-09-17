

#!/usr/bin/env python3
"""
Web音频处理应用
A web-based audio processing application
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import tempfile
import json
from audio_processor import AudioProcessor
import base64
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量存储处理器实例
processor = AudioProcessor()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传音频文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 保存上传的文件
        filename = file.filename
        file_path = os.path.join('/workspace', filename)
        file.save(file_path)
        
        # 加载音频文件
        if processor.load_audio(file_path):
            info = processor.get_audio_info()
            return jsonify({
                'success': True,
                'message': '文件上传成功',
                'filename': filename,
                'info': info
            })
        else:
            return jsonify({'error': '文件加载失败'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_audio():
    """处理音频"""
    try:
        action = request.json.get('action')
        params = request.json.get('params', {})
        
        if not action:
            return jsonify({'error': '没有指定处理动作'}), 400
        
        result = {'success': False, 'message': ''}
        
        if action == 'normalize':
            success = processor.normalize_audio(params.get('target_level', 0.8))
            result['message'] = '音频归一化完成' if success else '归一化失败'
            result['success'] = success
            
        elif action == 'filter':
            filter_type = params.get('type', 'low')
            cutoff_freq = params.get('cutoff_freq', 1000)
            order = params.get('order', 5)
            success = processor.apply_filter(filter_type, cutoff_freq, order)
            result['message'] = f'{filter_type}滤波器应用完成' if success else '滤波器应用失败'
            result['success'] = success
            
        elif action == 'volume':
            volume_factor = params.get('factor', 1.0)
            success = processor.change_volume(volume_factor)
            result['message'] = '音量调整完成' if success else '音量调整失败'
            result['success'] = success
            
        elif action == 'trim_silence':
            threshold = params.get('threshold', 0.01)
            success = processor.trim_silence(threshold)
            result['message'] = '静音去除完成' if success else '静音去除失败'
            result['success'] = success
            
        elif action == 'fade':
            fade_in = params.get('fade_in', 0.1)
            fade_out = params.get('fade_out', 0.1)
            success = processor.fade_in_out(fade_in, fade_out)
            result['message'] = '淡入淡出效果添加完成' if success else '淡入淡出效果添加失败'
            result['success'] = success
            
        elif action == 'features':
            features = processor.extract_features()
            if features:
                result['success'] = True
                result['message'] = '特征提取完成'
                result['features'] = features
            else:
                result['message'] = '特征提取失败'
                
        elif action == 'get_info':
            info = processor.get_audio_info()
            if info:
                result['success'] = True
                result['message'] = '信息获取完成'
                result['info'] = info
            else:
                result['message'] = '信息获取失败'
                
        else:
            result['message'] = f'未知的处理动作: {action}'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def save_audio():
    """保存音频文件"""
    try:
        filename = request.json.get('filename', 'processed_audio.wav')
        output_path = os.path.join('/workspace', filename)
        
        if processor.save_audio(output_path):
            return jsonify({
                'success': True,
                'message': '音频保存成功',
                'filename': filename,
                'download_url': f'/download/{filename}'
            })
        else:
            return jsonify({'error': '音频保存失败'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载处理后的音频文件"""
    try:
        file_path = os.path.join('/workspace', filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize_audio():
    """生成音频可视化图表"""
    try:
        plot_type = request.json.get('type', 'waveform')
        
        # 创建临时文件保存图表
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        success = False
        
        if plot_type == 'waveform':
            success = processor.plot_waveform(temp_path)
        elif plot_type == 'spectrum':
            success = processor.plot_spectrum(temp_path)
        elif plot_type == 'spectrogram':
            success = processor.plot_spectrogram(temp_path)
        
        if success:
            # 读取图片文件并转换为base64
            with open(temp_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'message': f'{plot_type}图表生成完成',
                'image_data': image_base64
            })
        else:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({'error': f'{plot_type}图表生成失败'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 确保模板目录存在
    os.makedirs('/workspace/templates', exist_ok=True)
    
    # 创建HTML模板
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频处理工具</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section h2 {
            color: #555;
            margin-top: 0;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select, button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background-color: #007bff;
            color: white;
            cursor: pointer;
            border: none;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .visualization {
            text-align: center;
            margin-top: 20px;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .features-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .features-table th, .features-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .features-table th {
            background-color: #f2f2f2;
        }
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .button-group button {
            flex: 1;
            min-width: 120px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频处理工具</h1>
        
        <!-- 文件上传部分 -->
        <div class="section">
            <h2>📁 上传音频文件</h2>
            <div class="form-group">
                <label for="audioFile">选择音频文件 (支持 WAV, MP3, FLAC 等格式):</label>
                <input type="file" id="audioFile" accept="audio/*">
            </div>
            <button onclick="uploadFile()">上传文件</button>
            <div id="uploadResult"></div>
        </div>
        
        <!-- 音频信息部分 -->
        <div class="section">
            <h2>📊 音频信息</h2>
            <button onclick="getAudioInfo()">获取音频信息</button>
            <div id="audioInfo"></div>
        </div>
        
        <!-- 音频处理部分 -->
        <div class="section">
            <h2>🔧 音频处理</h2>
            <div class="button-group">
                <button onclick="normalizeAudio()">音频归一化</button>
                <button onclick="applyLowPassFilter()">低通滤波</button>
                <button onclick="applyHighPassFilter()">高通滤波</button>
                <button onclick="trimSilence()">去除静音</button>
                <button onclick="addFadeEffect()">添加淡入淡出</button>
            </div>
            <div id="processResult"></div>
        </div>
        
        <!-- 音频特征部分 -->
        <div class="section">
            <h2>🔍 音频特征</h2>
            <button onclick="extractFeatures()">提取音频特征</button>
            <div id="featuresResult"></div>
        </div>
        
        <!-- 可视化部分 -->
        <div class="section">
            <h2>📈 音频可视化</h2>
            <div class="button-group">
                <button onclick="visualizeWaveform()">波形图</button>
                <button onclick="visualizeSpectrum()">频谱图</button>
                <button onclick="visualizeSpectrogram()">语谱图</button>
            </div>
            <div id="visualizationResult" class="visualization"></div>
        </div>
        
        <!-- 保存下载部分 -->
        <div class="section">
            <h2>💾 保存和下载</h2>
            <div class="form-group">
                <label for="outputFilename">输出文件名:</label>
                <input type="text" id="outputFilename" value="processed_audio.wav">
            </div>
            <button onclick="saveAudio()">保存音频</button>
            <div id="saveResult"></div>
        </div>
    </div>

    <script>
        let currentFile = null;

        async function uploadFile() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showResult('uploadResult', '请选择一个文件', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.success) {
                    currentFile = result.filename;
                    showResult('uploadResult', `文件上传成功: ${result.filename}`, 'success');
                    displayAudioInfo(result.info);
                } else {
                    showResult('uploadResult', result.error || '上传失败', 'error');
                }
            } catch (error) {
                showResult('uploadResult', `上传失败: ${error.message}`, 'error');
            }
        }

        async function getAudioInfo() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'get_info' })
                });

                const result = await response.json();
                
                if (result.success) {
                    displayAudioInfo(result.info);
                    showResult('audioInfo', '音频信息获取成功', 'success');
                } else {
                    showResult('audioInfo', result.message || '获取失败', 'error');
                }
            } catch (error) {
                showResult('audioInfo', `获取失败: ${error.message}`, 'error');
            }
        }

        async function normalizeAudio() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'normalize', params: { target_level: 0.8 } })
                });

                const result = await response.json();
                showResult('processResult', result.message, result.success ? 'success' : 'error');
            } catch (error) {
                showResult('processResult', `处理失败: ${error.message}`, 'error');
            }
        }

        async function applyLowPassFilter() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'filter', params: { type: 'low', cutoff_freq: 1000, order: 5 } })
                });

                const result = await response.json();
                showResult('processResult', result.message, result.success ? 'success' : 'error');
            } catch (error) {
                showResult('processResult', `处理失败: ${error.message}`, 'error');
            }
        }

        async function applyHighPassFilter() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'filter', params: { type: 'high', cutoff_freq: 1000, order: 5 } })
                });

                const result = await response.json();
                showResult('processResult', result.message, result.success ? 'success' : 'error');
            } catch (error) {
                showResult('processResult', `处理失败: ${error.message}`, 'error');
            }
        }

        async function trimSilence() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'trim_silence', params: { threshold: 0.01 } })
                });

                const result = await response.json();
                showResult('processResult', result.message, result.success ? 'success' : 'error');
            } catch (error) {
                showResult('processResult', `处理失败: ${error.message}`, 'error');
            }
        }

        async function addFadeEffect() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'fade', params: { fade_in: 0.2, fade_out: 0.2 } })
                });

                const result = await response.json();
                showResult('processResult', result.message, result.success ? 'success' : 'error');
            } catch (error) {
                showResult('processResult', `处理失败: ${error.message}`, 'error');
            }
        }

        async function extractFeatures() {
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ action: 'features' })
                });

                const result = await response.json();
                
                if (result.success) {
                    displayFeatures(result.features);
                    showResult('featuresResult', '特征提取成功', 'success');
                } else {
                    showResult('featuresResult', result.message || '特征提取失败', 'error');
                }
            } catch (error) {
                showResult('featuresResult', `特征提取失败: ${error.message}`, 'error');
            }
        }

        async function visualizeWaveform() {
            await visualizeAudio('waveform');
        }

        async function visualizeSpectrum() {
            await visualizeAudio('spectrum');
        }

        async function visualizeSpectrogram() {
            await visualizeAudio('spectrogram');
        }

        async function visualizeAudio(type) {
            try {
                const response = await fetch('/visualize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ type: type })
                });

                const result = await response.json();
                
                if (result.success) {
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + result.image_data;
                    img.alt = type + ' 图';
                    
                    const container = document.getElementById('visualizationResult');
                    container.innerHTML = '';
                    container.appendChild(img);
                    
                    showResult('visualizationResult', `${type} 图生成成功`, 'success');
                } else {
                    showResult('visualizationResult', result.error || '可视化失败', 'error');
                }
            } catch (error) {
                showResult('visualizationResult', `可视化失败: ${error.message}`, 'error');
            }
        }

        async function saveAudio() {
            const filename = document.getElementById('outputFilename').value;
            
            try {
                const response = await fetch('/save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ filename: filename })
                });

                const result = await response.json();
                
                if (result.success) {
                    const downloadLink = document.createElement('a');
                    downloadLink.href = result.download_url;
                    downloadLink.download = filename;
                    downloadLink.textContent = `点击下载 ${filename}`;
                    downloadLink.style.display = 'block';
                    downloadLink.style.marginTop = '10px';
                    
                    const container = document.getElementById('saveResult');
                    container.innerHTML = '';
                    container.appendChild(downloadLink);
                    
                    showResult('saveResult', '音频保存成功', 'success');
                } else {
                    showResult('saveResult', result.error || '保存失败', 'error');
                }
            } catch (error) {
                showResult('saveResult', `保存失败: ${error.message}`, 'error');
            }
        }

        function showResult(elementId, message, type) {
            const element = document.getElementById(elementId);
            element.innerHTML = `<div class="result ${type}">${message}</div>`;
        }

        function displayAudioInfo(info) {
            if (!info) return;
            
            let html = '<table class="features-table"><thead><tr><th>属性</th><th>值</th></tr></thead><tbody>';
            for (const [key, value] of Object.entries(info)) {
                html += `<tr><td>${key}</td><td>${value}</td></tr>`;
            }
            html += '</tbody></table>';
            
            document.getElementById('audioInfo').innerHTML = html;
        }

        function displayFeatures(features) {
            if (!features) return;
            
            let html = '<table class="features-table"><thead><tr><th>特征</th><th>值</th></tr></thead><tbody>';
            for (const [key, value] of Object.entries(features)) {
                html += `<tr><td>${key}</td><td>${typeof value === 'number' ? value.toFixed(6) : value}</td></tr>`;
            }
            html += '</tbody></table>';
            
            document.getElementById('featuresResult').innerHTML = html;
        }
    </script>
</body>
</html>
    """
    
    # 写入HTML模板文件
    with open('/workspace/templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("Web音频处理应用启动中...")
    print("访问地址: http://localhost:56274")
    app.run(host='0.0.0.0', port=56274, debug=True)

