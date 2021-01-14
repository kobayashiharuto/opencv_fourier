import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.chdir("C:/Users/owner/Desktop/opencv")
plt.axis("off")
# ここまで設定関連

# 画像を読み込む
img_before = cv2.imread('./original.jpg', 0)

# 画像データをフーリエ変換
dft = cv2.dft(np.float32(img_before), flags=cv2.DFT_COMPLEX_OUTPUT)
# 原点をずらして中心に持ってくる
dft_shifted = np.fft.fftshift(dft)
# 振幅スペクトルデータを作成
magnitude_spectrum = 20 * \
    np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

row, col = img_before.shape  # 画像のサイズ取得
row_mid = row//2  # 画像の中心のx座標
col_mid = col//2  # 画像の中心のy座標

# 通過領域（マスクをかけずに通す範囲）
range = 50

# マスクの作成
mask = np.zeros((row, col, 2), np.uint8)
mask[row_mid-range:row_mid+range, col_mid-range:col_mid+range] = 1

# マスクをかけたデータを作成する
masked_shifted = dft_shifted*mask
magnitude_spectrum_masked = 20 * \
    np.log(cv2.magnitude(masked_shifted[:, :, 0], masked_shifted[:, :, 1]))

# ずらしていたので元の位置に戻す
masked = np.fft.ifftshift(masked_shifted)

# マスクをかけたデータを逆フーリエ変換して画像データ作成
img_after = cv2.idft(masked)
img_after = cv2.magnitude(img_after[:, :, 0], img_after[:, :, 1])

# 振幅スペクトルを画像として保存
plt.imshow(magnitude_spectrum, cmap='gray')
plt.savefig('./magnitude.jpg', bbox_inches='tight', pad_inches=0)
# マスクをかけた振幅スペクトルを画像として保存
plt.imshow(magnitude_spectrum_masked, cmap='gray')
plt.savefig('./magnitude_masked.jpg', bbox_inches='tight', pad_inches=0)
# 読み込んだ元データを画像として保存
plt.imshow(img_before, cmap='gray')
plt.savefig('./img_before.jpg', bbox_inches='tight', pad_inches=0)
# マスクをかけた画像データを画像として保存
plt.imshow(img_after, cmap='gray')
plt.savefig('./img_after.jpg', bbox_inches='tight', pad_inches=0)
