import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data import load_and_process_data
from model import build_model

def train_and_evaluate():
    TEXT_MAXLEN = 50
    CTX_MAXLEN = 20
    BATCH_SIZE = 32
    EPOCHS = 10
    NUM_CLASSES = 3

    X_t_tr, X_t_te, X_c_tr, X_c_te, y_tr, y_te, tokenizer = load_and_process_data(
        'sentiment_data.csv', TEXT_MAXLEN, CTX_MAXLEN
    )

    configs = [
        (True, False, 'pretrained+concat'),
        (True, True, 'pretrained+cross_attn'),
        (False, False, 'scratch+concat'),
        (False, True, 'scratch+cross_attn')
    ]

    results = {}
    histories = []
    models = {}

    for use_pretrained, use_cross_attention, name in configs:
        print(f"\n>> Experiment: {name}")
        model = build_model(
            tokenizer, use_pretrained, use_cross_attention, TEXT_MAXLEN, CTX_MAXLEN, NUM_CLASSES
        )
        history = model.fit(
            [X_t_tr, X_c_tr], y_tr,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        histories.append(history.history)
        preds = np.argmax(model.predict([X_t_te, X_c_te]), axis=1)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='macro')
        print(f"{name:12s} → acc={acc:.4f}, f1_macro={f1:.4f}")
        print(classification_report(y_te, preds, digits=4))
        results[name] = (acc, f1)
        models[name] = model

    plot_histories(histories, [name for _, _, name in configs])

    names = list(results.keys())
    accs = [results[n][0] for n in names]
    f1s = [results[n][1] for n in names]
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width/2, accs, width, label='Accuracy')
    bars2 = plt.bar(x + width/2, f1s, width, label='Macro-F1')

    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Model Comparison: Test Accuracy vs Macro-F1')
    plt.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    return models, tokenizer, results

def plot_histories(histories, names):
    plt.figure(figsize=(12, 5))
    for history, name in zip(histories, names):
        plt.plot(history['accuracy'], label=f'{name} train_acc')
        plt.plot(history['val_accuracy'], label=f'{name} val_acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for history, name in zip(histories, names):
        plt.plot(history['loss'], label=f'{name} train_loss')
        plt.plot(history['val_loss'], label=f'{name} val_loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def predict_sentiment(models, tokenizer, text, context, text_maxlen=50, ctx_maxlen=20):
    from data import clean_and_truncate
    text_proc = clean_and_truncate(text, text_maxlen)
    context_proc = clean_and_truncate(context, ctx_maxlen)

    text_encodings = tokenizer([text_proc], padding='max_length', truncation=True, max_length=text_maxlen, return_tensors='tf')
    context_encodings = tokenizer([context_proc], padding='max_length', truncation=True, max_length=ctx_maxlen, return_tensors='tf')
    X_text = text_encodings['input_ids']
    X_context = context_encodings['input_ids']

    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predictions = {}
    for name, model in models.items():
        pred = np.argmax(model.predict([X_text, X_context], verbose=0), axis=1)[0]
        predictions[name] = label_map[pred]
    return predictions

if __name__ == "__main__":
    models, tokenizer, results = train_and_evaluate()

    print("\nEnter text and context to predict sentiment (type 'quit' to exit).")
    while True:
        text = input("Enter text: ")
        if text.lower() == 'quit':
            break
        context = input("Enter context: ")
        if context.lower() == 'quit':
            break

        predictions = predict_sentiment(models, tokenizer, text, context)
        print("\nPredictions:")
        for name, pred in predictions.items():
            print(f"  {name:12s} → {pred}")
        print()