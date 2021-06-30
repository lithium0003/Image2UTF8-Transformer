フォルダ構成
    enfont/          英語用フォントフォルダ
    jpfont/　　　　　　日本語用フォントフォルダ
    handwritten/     手書き文字画像フォルダ
    load_font/       fontのビットマップを取得するプログラム
    codepoints.csv   青空文庫の外字をUnicodeに変換するテーブル
    id_map.csv       学習用文字リスト


kanji_list.txt
教育漢字と常用漢字のリスト

other_list.txt
記号のリスト

2nd_kanji.txt
第2水準漢字

3rd_kanji.txt
第3水準漢字

4th_kanji.txt
第4水準漢字

get_kanji_list.py
漢字一覧のサイト http://www13.plala.or.jp/bigdata から
各水準の漢字リスト(1st-4th_kanji.txt)を作成

make_glyphid.py
id_map.csvを生成する

load_font/load_font
fontのビットマップを得る
Usage: load_font/load_font font_path size unicode

英語フォントリスト.txt
enfont/ フォルダに配置すべきフォントの取得方法の一例を示す

日本語フォントリスト.txt
jpfont/ フォルダに配置すべきフォントの取得方法の一例を示す

漢字一覧の1面.txt
漢字一覧の2面.txt
非漢字一覧.txt
Wikipediaより取得した、jisコードとunicodeの対応csvを作る為の元データ
codepoints.csvは、この3ファイルから 面区点 と Unicode カラムを抽出して作成
