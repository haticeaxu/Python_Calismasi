import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

file_path = "/Users/haticeaksu/Desktop/Yüksek Lisans/Makine Öğrenmesi/vize/veri-seti.txt"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = [
    "Hamilelik Sayısı", "Glukoz", "Kan Basıncı", "Cilt Kalınlığı", "İnsülin",
    "Vücut Kitle İndeksi (BMI)", "Diyabet Soy Ağacı Fonksiyonu", "Yaş", "Sonuç (0: Yok, 1: Var)"
]


X = data.drop(columns=["Sonuç (0: Yok, 1: Var)"])
y = data["Sonuç (0: Yok, 1: Var)"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)


mlr_coefficients = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": mlr_model.coef_})
print("Çoklu Doğrusal Regresyon Katsayıları:")
print(mlr_coefficients)


plt.figure(figsize=(10, 6))
sns.barplot(x=mlr_coefficients["Feature"], y=mlr_coefficients["Coefficient"], palette="viridis")
plt.title("Çoklu Doğrusal Regresyon Katsayıları")
plt.xlabel("Özellikler")
plt.ylabel("Katsayı Değeri")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


mlr_model = LogisticRegression(max_iter=1000)
mlr_model.fit(X_train, y_train)


y_pred = mlr_model.predict(X_test)


print("\nMultinominal Lojistik Regresyon Doğruluk Oranı:",
      accuracy_score(y_test, y_pred))

# Sonuçlar
print("\nRaporlama:")
print(classification_report(y_test, y_pred))
print("\nKonfüzyon Matrisi:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=mlr_model.coef_.flatten(), palette="viridis")
plt.title("Multinominal Lojistik Regresyon Katsayıları")
plt.xlabel("Özellikler")
plt.ylabel("Katsayı Değeri")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Model performansı
print("\nRaporlama:")
print(classification_report(y_test, y_pred))
print("\nKonfüzyon Matrisi:")
print(confusion_matrix(y_test, y_pred))

# Konfüzyon Matris
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="viridis")
plt.title("Konfüzyon Matrisi")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.show()

# Tahmin
y_prob = mlr_model.predict_proba(X_test)[:, 1]

# ROC eğrisi oluştur
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ROC eğrisi
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()






