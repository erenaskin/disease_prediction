import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

#%% Veri yükleme ve ön işleme

def load_and_preprocess_data(filename):
    """Veri yükleme ve ön işleme fonksiyonu"""
    data = pd.read_csv(filename)
    
    print(f"Toplam Örnek Sayısı: {data.shape[0]}")
    print(f"Özellik Sayısı: {data.shape[1]}")

    if data.isnull().any().any():
        print("Eksik değerler tespit edildi.")
        print(data.isnull().sum())
    else:
        print("Eksik değer bulunmamaktadır.")

    print("Sınıf Dağılımı:")
    print(data['prognosis'].value_counts())

    return data

#%% Model eğitimi fonksiyonu
def train_model(X_train, y_train, X_test, y_test):
    """Model eğitim fonksiyonu"""
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=50,
        random_state=104,
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        use_best_model=True
    )

    return model

#%% Değerlendirme fonksiyonu

def evaluate_model(model, X_test, y_test, label_encoder):
    """Modeli değerlendirme ve rapor oluşturma"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nSınıf Bazlı Rapor:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))  # Daha okunabilir bir boyut
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.show()

#%% Özellik önemliliği görselleştirme

def plot_feature_importance(model, feature_names):
    """En önemli 20 özelliği görselleştirir"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('En Önemli 20 Özellik')
    plt.xlabel('Özellik Önemi')
    plt.ylabel('Özellikler')
    plt.tight_layout()
    plt.show()

#%% Ana çalıştırma fonksiyonu

def main():
    filename = "symbipredict_2022.csv"
    data = load_and_preprocess_data(filename)

    X = data.drop(columns=['prognosis'])
    y = data['prognosis']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.values.ravel())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=104, stratify=y
    )

    model = train_model(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test, label_encoder)
    plot_feature_importance(model, X.columns)

    joblib.dump(model, 'best_model.pkl')
    joblib.dump(X.columns.tolist(), 'symptoms.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

#%% Programı çalıştır

if __name__ == "__main__":
    main()
