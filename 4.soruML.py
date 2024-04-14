
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


file_path = "/Users/haticeaksu/Desktop/Yüksek Lisans/Makine Öğrenmesi/vize/veri-seti.txt"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = [
    "Hamilelik Sayısı", "Glukoz", "Kan Basıncı", "Cilt Kalınlığı", "İnsülin",
    "Vücut Kitle İndeksi (BMI)", "Diyabet Soy Ağacı Fonksiyonu", "Yaş", "Sonuç (0: Yok, 1: Var)"
]


X = data.drop(columns=["Sonuç (0: Yok, 1: Var)"])
y = data["Sonuç (0: Yok, 1: Var)"]

# Veriyi eğitim ve test alt kümelerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


y_pred = dt_model.predict(X_test)


print("Karar Ağacı Sınıflandırma Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Karar ağacı göreseli
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=["0", "1"], filled=True)
plt.title("Karar Ağacı")
plt.show()


# Model performansı
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Konfüzyon matris
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Konfüzyon Matrisi")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.show()


print("Sınıflandırma Raporu:")
print(class_report)

y_prob = dt_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
