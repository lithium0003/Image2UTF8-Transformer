# Image2UTF8-Transformer
このレポジトリは、文字の画像をUTF8エンコード文字列に変換するTransformerモデルを学習させるコードです。

![model](https://user-images.githubusercontent.com/4783887/123975754-8ddb2c00-d9f8-11eb-8e4d-159c669c81d2.png)

## Description
128x128x3のカラー画像として文字を入力します。
この画像列を、EfficientNetに入れ、各画像256次元の表現を得ます。
256xNの入力データを、256xMの出力データにTransformerモデルを用いて変換します。
出力データは、MバイトのUTF8エンコード文字列になります。

推論は、Mask-Predictの方法で高速化しています。(T=4,L=4)

## Requrement
学習元データとして次のファイルを用意します
+ 日本語フォントファイル
+ 英語フォントファイル
+ 手書き文字セット

次のライブラリに依存します
- freetype
- tensorflow
- umap-learn
- tqdm
- matplotlib
- pillow

CoreMLモデルに変換する場合は追加で必要
- coremltools

## Setup
data/load_font フォルダにあるコードをコンパイルします。
``` bash
sudo apt install libfreetype-dev
cd data/load_font
make
```

python環境に必要モジュールをインストールします。
``` bash
python3 -m venv envs/tensorflow
. envs/tensorflow/bin/activate
pip install wheel
pip install tensorflow
pip install umap-learn
pip install tqdm
pip install matplotlib
```

## Prepare
学習データの準備をします。
- data/日本語フォントリスト.txt を参考に日本語フォントファイルを取得し、 data/jpfont フォルダに配置します。
- data/英語フォントリスト.txt　を参考に英語フォントファイルを取得し、 data/enfont フォルダに配置します。
- 手書き文字セットを、　data/handwritten に配置します。サンプルとして、 data/handwritten.tar.gzが用意されていますので、展開して用いるか、 https://apps.apple.com/jp/app/id1569208844 などのアプリを用いて独自の手書き文字セットを作成します。
- 学習する文字セットは、 data/id_map.csv で指定します。既に作成してありますが、変更したい場合は、data/make_glyphid.py を参考に作成し直してください。

## Train
### 1st Step
最初に、前段のEfficientNetを学習させます。
ランダムに、文字画像を与えて、104x100=10400文字を識別できるように、256次元の表現を出せるモデルを構築します。

``` bash
python3 train.py
```

学習が終わると、　result/encoder にモデルが作成されます。

## 2nd Step
次に、後段のTransformerを学習させます。
32文字以下の文字画像を、4x32=128byte以下のUTF8文字列(256分類で1文字で1byte出力)に変換される学習をします。
ランダムに、登録した文字からランダム文字列、ランダムな日付や数詞表現、青空文庫からランダムに作品を取得、Wikipediaからランダムに記事を取得、の文字列を作成し、そこから32文字以下のランダム文字列画像を生成し、UTF8で文字列が一致するように学習させます。

``` bash
python3 train2.py
```

学習が終わると、result/transformer_weights　に重みが出力されます。

## iOSでのモデル使用
iOSで学習済みモデルを使用できます。
以下のコマンドで変換できます。

``` bash
python3 convert_model.py
```

## Reference
- https://qiita.com/halhorn/items/c91497522be27bde17ce
- https://www.tensorflow.org/tutorials/text/transformer
- https://arxiv.org/abs/1904.09324



