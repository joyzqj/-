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
app = Flask(__name__)
CORS(app)

try:
    with open('api_config.json', 'r') as config_file:
        api_config = json.load(config_file)
except Exception as e:
    logging.error(f"Error loading API configuration: {e}")

model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'url' not in request.form:
            return "No URL provided", 400

        url = request.form['url']
        if request.form.get('type') == "content":  
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                file_like_object = io.BytesIO(response.content)
                segments, info = model.transcribe(file_like_object, beam_size=5, language="zh", initial_prompt="转录为简体中文，这是一段会议记录。", vad_filter=True,    vad_parameters=dict(min_silence_duration_ms=500),prepend_punctuations=False,append_punctuations=False)
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
                client = client = OpenAI(api_key=api_config["openai"]["api_key"],base_url=api_config["openai"]["base_url"]) 
                messages = [
                    {"role": "system", "content": "严格按照摘要提取的要求对下列文本做摘要提取，参考下列示例文档，输出时直接输出提取的摘要内容不要前面的‘摘要：’：‘男人惊恐地看着前方,赶紧将头埋进水里,全程屏住呼吸不敢动弹。紧接着一双巨大的大脚出现在眼前,脚下的绵羊赶紧逃跑,谁知下一秒就被一只大手抓住。此时埋进水中的男人依旧不敢动弹,下一秒眼前恐怖的一幕直接把他吓呆了,害怕的男人惊恐地张大了嘴巴。躲在树后的同伴尝出了一口气之后,忍不住好奇地他探出头去观望,他不可思议地看着眼前的庞然大雾。然而就在他看的入神时,巨人好像感受到了他的气味,害怕的小胖赶紧躲回树后,紧接着巨人大步往前走,再也憋不住的男人赶紧冒出水里。可怕巨人还是发现了树后的小胖,只见他直接将整个大树连根拔起,顿感不妙的小胖连连后腿,两个人就这样深情地对视了三年之久。反应过来的小胖这才想起转身往后逃跑,巨人就这样呆呆地看着眼前急速奔跑的小胖,丝毫不慌任停他跑,紧接着直接三步并乘两步一掌就将他拍晕过去。对于没见过的人类,可怕的巨人直接凑上去仔细观察。就在他抓起小胖准备离开时,救人心切的队长直接持剑冲了上去,猛的将力剑刺入巨人的脚踝,吃痛的巨人往后看了一眼,下一秒只是轻轻一脚,队长就砸在石头上晕了过去,紧接着巨人把脚上的力剑拔掉,这下好了,两个人都被巨人直接打包带走,一旁的捷克只能眼睁睁地看着,同伴就这样落入巨人之手,巨人看了看手中的猎物很深埋迷,接着转身离开,害怕了巨人,在他被发现的捷克赶紧再次躲回水面,庆幸的是巨人并没有发现他,就这样从他身旁路走。等巨人走远后,再悄悄地跟在他的身后,再寻找机会拯救同伴,与此同时,其他三人在途径一处悬崖时,卷毛哥假装看到公主,吸引同伴光头过来,然后趁机不备将他推下万丈巡逻,这样他就可以为所欲为了,然而就在他们为奸计得逞得意洋洋失落。一只大手就将疯屁男抓了起来,他话不说就直接塞进铁门,卷毛哥整个人都吓得压麻呆嘴,完全没意识到,更可怕的危险正咸服在身后。’->：\
                    \
    一群男人遭遇了巨人的袭击。一个男人因恐惧而将头埋入水中，屏住呼吸。巨人出现，抓住了逃跑的绵羊。男人在水中目睹了恐怖一幕，而他的同伴则因好奇探头观望，被巨人发现。巨人将树连根拔起，小胖逃跑但最终被巨人拍晕。队长试图用剑攻击巨人，但同样被击败。巨人带走了小胖和队长。捷克未被发现，计划跟踪巨人以救出同伴。同时，卷毛哥试图推同伴下悬崖，却被巨人抓住，暗示了更大的危险。  "},
                    {"role": "user", "content": file_content.decode('utf-8')},
                ]
                response_gen = client.chat.completions.create(model="moonshot-v1-128k", messages=messages, temperature=0.1, stream=True)
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=6006, debug=False)






























