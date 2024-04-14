import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


file_path = "/Users/haticeaksu/Desktop/Yüksek Lisans/Makine Öğrenmesi/vize/veri-seti.txt"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = [
    "Hamilelik Sayısı", "Glukoz", "Kan Basıncı", "Cilt Kalınlığı", "İnsülin",
    "Vücut Kitle İndeksi (BMI)", "Diyabet Soy Ağacı Fonksiyonu", "Yaş", "Sonuç (0: Yok, 1: Var)"
]


X = data.drop(columns=["Sonuç (0: Yok, 1: Var)"])
y = data["Sonuç (0: Yok, 1: Var)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


y_pred = nb_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


y_prob = nb_model.predict_proba(X_test)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_prob)

# ROC eğrisi
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes Classifier')
plt.legend()
plt.grid(True)
plt.show()

# Konfüzyon matris
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


print("Classification Report:")
print(class_report)

