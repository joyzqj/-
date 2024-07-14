# import requests
# from werkzeug.utils import secure_filename
# from flask import Flask, request, Response
# from faster_whisper import WhisperModel
# from flask_cors import CORS
# import io

# app = Flask(__name__)
# CORS(app)
# model_size = "medium"
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# @app.route('/transcribe', methods=['POST'])
# def transcribe():
#     # 检查请求中是否包含URL字段
#     if 'url' not in request.form:
#         return "No URL provided", 400

#     # 获取MP3文件的URL
#     url = request.form['url']
#     try:
#         # 下载MP3文件
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # 确保请求成功

#         # 使用io.BytesIO来模拟文件对象
#         file_like_object = io.BytesIO(response.content)

#         # 使用 faster_whisper 进行转录
#         segments, info = model.transcribe(file_like_object, beam_size=5)

#         def generate():
#             # 输出检测到的语言信息
#             # yield f"Detected language '{info.language}' with probability {info.language_probability}\n"
#             # 逐段输出文本，不包含时间戳
#             for segment in segments:
#                 yield f"{segment.text}"

#         # 使用 Response 来确保生成器在请求上下文中运行
#         result= Response(generate(), mimetype='text/plain')
#         return result

#     except requests.RequestException as e:
#         return str(e), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6006, debug=True)


from fastapi import FastAPI, BackgroundTasks, FileResponse, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import requests
from io import BytesIO
from pydantic import BaseModel
from faster_whisper import WhisperModel
import json
from openai import OpenAI
import torch

app = FastAPI()

# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_size = "medium"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperModel(model_size, device=device, compute_type="float16")

client = OpenAI(api_key="sk-xNT3N9faAuvKchCW9vGOOo1X2xadqZy9W1SUrgjnixK1Nb9U", base_url="https://api.moonshot.cn/v1")

class TranscriptionRequest(BaseModel):
    url: str
    type: str

@app.post("/transcribe")
async def transcribe(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    url = data.get("url")
    type = data.get("type")
    
    if not url:
        return Response(content="No URL provided", status_code=400)

    try:
        if type == "content":
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_like_object = BytesIO(response.content)
            segments, info = model.transcribe(file_like_object, beam_size=5, language="zh", initial_prompt="转录为简体中文", vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
            return Response(content=generate(segments), media_type='text/plain')

        elif type == "tips":
            content = data.get("content")
            file_content = content
            messages = [
                {"role": "system", "content": "对文本做摘要提取,摘要内容必须精准、全面的概括文本内容"},
                {"role": "user", "content": file_content},
            ]
            response_gen = client.chat.completions.create(model="moonshot-v1-128k", messages=messages, temperature=0.2, stream=True)
            background_tasks.add_task(generate1, response_gen)
            return Response(content="Transcription started", status_code=202)

    except requests.RequestException as e:
        return Response(content=str(e), status_code=500)

def generate(segments):
    for segment in segments:
        yield f"{segment.text}"

async def generate1(response_gen):
    try:
        for chunk in response_gen:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                yield delta_content.encode('utf-8')
    except Exception as e:
        print(f"Error generating response: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6006, log_level="info")

