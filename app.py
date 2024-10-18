import gradio as gr
import groq
import os
import io
import numpy as np
import soundfile as sf
from PIL import Image
import base64

# 環境変数からGROQ_API_KEYを取得
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY環境変数を設定してください。")

# Groqクライアントの初期化
client = groq.Client(api_key=api_key)

def transcribe_audio(audio):
    if audio is None:
        return "音声が提供されていません。", ""
    sr, y = audio

    # ステレオの場合はモノラルに変換
    if y.ndim > 1:
        y = y.mean(axis=1)

    # 音声の正規化
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # 音声をバッファに書き込み
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format='wav')
    buffer.seek(0)

    try:
        # Whisper大規模モデルを使用して文字起こし
        completion = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.wav", buffer),
            response_format="text"
        )
        transcription = completion
    except Exception as e:
        transcription = f"文字起こしエラー: {str(e)}"

    response = generate_response(transcription)
    return transcription, response

def generate_response(transcription):
    if not transcription or transcription.startswith("エラー"):
        return "有効な文字起こしがありません。もう一度話してください。"

    try:
        # Llama 3.1 70Bモデルを使用してテキスト生成
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "あなたは役立つアシスタントです。"},
                {"role": "user", "content": transcription}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"応答生成エラー: {str(e)}"

def analyze_image(image):
    if image is None:
        return "画像がアップロードされていません。", None

    # numpy配列をPIL Imageに変換
    image_pil = Image.fromarray(image.astype('uint8'), 'RGB')

    # PIL画像をbase64に変換
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        # Llama 3.2 11B Visionモデルを使用して画像を分析
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "この画像を詳細に説明してください。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )
        description = chat_completion.choices[0].message.content
    except Exception as e:
        description = f"画像分析エラー: {str(e)}"

    return description

def respond(message, chat_history):
    if chat_history is None:
        chat_history = []

    # APIのメッセージ履歴を準備
    messages = []
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    try:
        # Llama 3.1 70Bモデルを使用してアシスタントの応答を生成
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
        )
        assistant_message = completion.choices[0].message.content
        chat_history.append((message, assistant_message))
    except Exception as e:
        assistant_message = f"エラー: {str(e)}"
        chat_history.append((message, assistant_message))

    return "", chat_history, chat_history  # 状態を3番目の出力として返す

# Groqバッジとカラースキーム用のカスタムCSS
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.gr-button-primary {
    background-color: #f55036 !important;
    border-color: #f55036 !important;
}
#groq-badge {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎙️ Groq x Gradio マルチモーダル Llama-3.2 および Whisper")

    with gr.Tab("音声"):
        gr.Markdown("## AIと会話する")
        with gr.Row():
            audio_input = gr.Audio(type="numpy", label="話すか音声をアップロード")
        with gr.Row():
            transcription_output = gr.Textbox(label="文字起こし")
            response_output = gr.Textbox(label="AIアシスタントの応答")
        process_button = gr.Button("処理", variant="primary")
        process_button.click(
            transcribe_audio,
            inputs=audio_input,
            outputs=[transcription_output, response_output]
        )

    with gr.Tab("画像"):
        gr.Markdown("## 分析用の画像をアップロード")
        with gr.Row():
            image_input = gr.Image(type="numpy", label="画像をアップロード")
        with gr.Row():
            image_description_output = gr.Textbox(label="画像の説明")
        analyze_button = gr.Button("画像を分析", variant="primary")
        analyze_button.click(
            analyze_image,
            inputs=image_input,
            outputs=[image_description_output]
        )

    with gr.Tab("チャット"):
        gr.Markdown("## AIアシスタントとチャット")
        chatbot = gr.Chatbot()
        state = gr.State([])  # チャットの状態を初期化
        with gr.Row():
            user_input = gr.Textbox(show_label=False, placeholder="ここにメッセージを入力...", container=False)
            send_button = gr.Button("送信", variant="primary")
        send_button.click(
            respond,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state],
        )

    # Groqバッジを追加
    gr.HTML("""
    <div id="groq-badge">
        <div style="color: #f55036; font-weight: bold;">GROQ提供</div>
    </div>
    """)

    gr.Markdown("""
    ## このアプリの使い方:

    ### 音声タブ
    1. マイクアイコンをクリックして話すか、音声ファイルをアップロードします。
    2. "処理"ボタンをクリックして、音声を文字起こしし、AIアシスタントから応答を生成します。
    3. 文字起こしとAIアシスタントの応答が、それぞれのテキストボックスに表示されます。

    ### 画像タブ
    1. 画像アップロードエリアをクリックして画像をアップロードします。
    2. "画像を分析"ボタンをクリックして、画像の詳細な説明を取得します。
    3. アップロードされた画像とその説明が表示されます。

    ### チャットタブ
    1. 下部のテキストボックスにメッセージを入力します。
    2. "送信"ボタンをクリックしてAIアシスタントと対話します。
    3. 会話がチャットインターフェースに表示されます。
    """)

demo.launch()
