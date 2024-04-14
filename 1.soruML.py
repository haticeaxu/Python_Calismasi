

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

file_path = "/Users/haticeaksu/Desktop/Yüksek Lisans/Makine Öğrenmesi/vize/veri-seti.txt"

column_names = ["Number of times pregnant", "Plasma glucose concentration", "Diastolic blood pressure", "Triceps skinfold thickness", "2-Hour serum insulin", "Body mass index", "Diabetes pedigree function", "Age", "Class variable"]

df = pd.read_csv(file_path, sep="\t", names=column_names)

print(df.describe())


plt.figure(figsize=(20, 10))
for i, col in enumerate(df.columns):
    if i < 8:  # 2 satır ve 4 sütun için maksimum 8 subplot oluşturulabilir
        plt.subplot(2, 4, i+1)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(col)
        plt.xlabel("Değer")
        plt.ylabel("Frekans")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 8))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.xlabel("Değer")
    plt.ylabel("Sütun")
plt.tight_layout()
plt.show()


# Z-Skoru Yöntemi
z_scores = df.apply(lambda x: (x - x.mean()) / x.std())
df_cleaned_z = df[(z_scores < 3).all(axis=1)]


# print(df_cleaned_z)


# IQR Yöntemi
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_cleaned_iqr = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# print(df_cleaned_iqr)

plt.figure(figsize=(20, 10))
num_plots = min(len(df_cleaned_z.columns), 8)  # Maksimum 8 subplot oluştur
for i in range(num_plots):
    plt.subplot(2, 4, i+1)
    sns.histplot(df_cleaned_z.iloc[:, i], bins=20, kde=True)
    plt.title(df_cleaned_z.columns[i])
    plt.xlabel("Değer")
    plt.ylabel("Frekans")
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 10))
num_plots = min(len(df_cleaned_iqr.columns), 8)  # Maksimum 8 subplot oluştur
for i in range(num_plots):
    plt.subplot(2, 4, i+1)
    sns.histplot(df_cleaned_z.iloc[:, i], bins=20, kde=True)
    plt.title(df_cleaned_z.columns[i])
    plt.xlabel("Değer")
    plt.ylabel("Frekans")
plt.tight_layout()
plt.show()


# Boxplot görselleştirmesi
plt.figure(figsize=(8, 8))
for i, col in enumerate(df_cleaned_iqr.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.xlabel("Değer")
    plt.ylabel("Sütun")
plt.tight_layout()
plt.show()


# Boxplot görselleştirmesi
plt.figure(figsize=(8, 8))
for i, col in enumerate(df_cleaned_z.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.xlabel("Değer")
    plt.ylabel("Sütun")
plt.tight_layout()
plt.show()



# Min-Max normalizasyonu
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)


# print(normalized_data)

