# Portable LSFM Spot Viewer

`portable_spot_viewer.ipynb` をJupyterで開き、上から順に実行してください。
Notebookは同じフォルダの`manifest.csv`、`raw/`、`postprocessed/`、
`spots/`を自動的に参照します。

## 別PCでの準備

```powershell
python -m pip install -r portable_viewer_requirements.txt
python -m jupyter lab
```

## データ形式

- `raw/<dataset>.zip`: Raw画像180枚のgrayscale JPEG
- `postprocessed/<dataset>.zip`: 前処理画像180枚のgrayscale JPEG
- `spots/<dataset>.npz`: Z順に格納したX/Y/Z座標
- `manifest.csv`: datasetと各ファイルの対応表

X/Yは0.1 pixel単位で量子化しています。MAY08以外には、
Ld・Lv・Rd・Rvごとの全例和集合脳マスクを適用しています。
MAY08はマスク未適用です。

画像は8-bit JPEGですが、Viewer上でBlack/White pointとGammaを調整できます。
Gammaを1より大きくすると中間調が明るくなり、1より小さくすると暗くなります。
