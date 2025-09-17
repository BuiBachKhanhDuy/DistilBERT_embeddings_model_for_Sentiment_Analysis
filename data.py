import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

def clean_and_truncate(s: str, max_words: int) -> str:
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    words = s.split()[:max_words]
    return ' '.join(words)

def load_and_process_data(file_path, text_maxlen=50, ctx_maxlen=20):
    df = pd.read_csv(file_path)
    print("Raw data shape:", df.shape)

    df['text_proc'] = df['text'].apply(lambda t: clean_and_truncate(t, text_maxlen))
    df['context_proc'] = df['context'].apply(lambda c: clean_and_truncate(c, ctx_maxlen))

    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label_enc'] = df['label'].map(label_map)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    text_encodings = tokenizer(df['text_proc'].tolist(), padding='max_length', truncation=True, max_length=text_maxlen, return_tensors='tf')
    context_encodings = tokenizer(df['context_proc'].tolist(), padding='max_length', truncation=True, max_length=ctx_maxlen, return_tensors='tf')

    X_text = text_encodings['input_ids']
    X_context = context_encodings['input_ids']
    y = df['label_enc'].values

    X_t_tr, X_t_te, X_c_tr, X_c_te, y_tr, y_te = train_test_split(
        X_text.numpy(), X_context.numpy(), y,
        test_size=0.2, shuffle=True, random_state=42
    )

    print("Shapes:")
    print("  X_text:", X_text.shape, "X_context:", X_context.shape)
    print("  Train:", X_t_tr.shape, X_c_tr.shape, y_tr.shape)
    print("  Test: ", X_t_te.shape, X_c_te.shape, y_te.shape)

    return X_t_tr, X_t_te, X_c_tr, X_c_te, y_tr, y_te, tokenizer