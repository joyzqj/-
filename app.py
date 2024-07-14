import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, Response
from faster_whisper import WhisperModel
from flask_cors import CORS
import io
import logging
import json
from openai import OpenAI
import torch
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# 使用线程池来管理并发请求
executor = ThreadPoolExecutor(max_workers=5)  # 可以根据需要调整线程池大小

try:
    with open('api_config.json', 'r') as config_file:
        api_config = json.load(config_file)
except Exception as e:
    logging.error(f"Error loading API configuration: {e}")

model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    def handle_transcription():
        try:
            if 'url' not in request.form:
                return "No URL provided", 400

            url = request.form['url']
            if request.form.get('type') == "content":  
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    file_like_object = io.BytesIO(response.content)
                    segments, info = model.transcribe(file_like_object, beam_size=5, language="zh", initial_prompt="转录为简体中文，这是一段会议记录。", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),prepend_punctuations=False,append_punctuations=False)
                    def generate(segments):
                        for segment in segments:
                            yield f"{segment.text}"
                    return Response(generate(segments), mimetype='text/plain')
                except requests.RequestException as e:
                    return str(e), 500
                except Exception as e:
                    logging.error(f"Error processing transcription: {e}")
                    return "Error processing transcription", 500

            elif request.form.get('type') == "tips": 
                result = request.form.get('content')
                if not result:
                    return "No content provided", 400
                try:
                    file_content = result.encode('utf-8')
                    client = OpenAI(api_key=api_config["openai"]["api_key"],base_url=api_config["openai"]["base_url"]) 
                    messages = [
                        {"role": "system", "content": "请提供一个包含核心观点和主要论点的摘要。字数占总文本的10%"},
                        {"role": "user", "content": file_content.decode('utf-8')},
                    ]
                    response_gen = client.chat.completions.create(model="moonshot-v1-128k", messages=messages, temperature=0.2, stream=True)
                    def generate1(response_gen):
                        try:
                            for chunk in response_gen:
                                delta_content = chunk.choices[0].delta.content
                                if delta_content is not None:
                                    yield delta_content.encode('utf-8')
                        except Exception as e:
                            logging.error(f"Error generating response: {e}")
                    return Response(generate1(response_gen), mimetype='text/plain')
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    return "Error generating response", 500

        except Exception as e:
            logging.error(f"Unhandled exception: {e}")
            return "An unexpected error occurred", 500

    # 使用线程池执行处理函数
    future = executor.submit(handle_transcription)
    return Response(future.result(), mimetype='text/plain')  # 将future的结果作为响应返回

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6006, debug=False)