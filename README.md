
# GradioとGroqを活用したマルチモーダルアプリケーションの構築ブロック

[ビデオデモ](https://github.com/user-attachments/assets/0ab0f71a-4b0a-4d58-ae79-02573aa8a21d)

このリポジトリには、GradioとGroqを使用して高速なマルチモーダルアプリケーションを構築する方法を示すアプリケーションが含まれています。具体的には、WhisperとLlama-3.2-visionを使用して、音声からテキスト、テキストからLLMの応答、画像からテキスト、そして従来のチャット機能を実現しています。

### クイックスタート

Gradioアプリを実行するには、以下の手順に従ってください：

~~~
python3 -m venv venv
~~~

~~~
source venv/bin/activate
~~~

~~~
pip3 install -r requirements.txt
~~~

~~~
export GROQ_API_KEY=gsk...
~~~

~~~
python3 app.py
~~~

これでアプリが http://127.0.0.1:7860 でホストされます！

### UVを使用する場合の設定方法

UVを使用する場合は、以下の手順で環境を設定できます：

1. 仮想環境の作成：
   ```
   uv venv
   ```

2. 仮想環境の有効化（Windows PowerShellの場合）：
   ```
   .venv\Scripts\activate
   ```

3. 依存関係のインストール：
   ```
   uv pip install -r requirements.txt
   ```

注意：UVを使用する場合、`pip3`ではなく`pip`サブコマンドを使用します。

これらの手順を実行後、アプリケーションを通常通り起動できます。
