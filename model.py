from tensorflow.keras.layers import Input, Dense, Layer, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from transformers import TFDistilBertModel

class PositionalEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerEncoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        for layer in self.enc_layers:
            x = layer(x, training)
        return self.layernorm(x)

def build_model(tokenizer, use_pretrained, use_cross_attention, text_maxlen, ctx_maxlen, num_classes=3,
                embedding_dim=100, num_layers=2, num_heads=4, dff=128, dropout=0.1):
    vocab_size = tokenizer.vocab_size
    text_input = Input(shape=(text_maxlen,), dtype=tf.int32, name="text_input")
    context_input = Input(shape=(ctx_maxlen,), dtype=tf.int32, name="context_input")

    if use_pretrained:
        distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        distilbert.trainable = False
        text_out = distilbert(text_input)[0]
        context_out = distilbert(context_input)[0]
        text_cls = text_out[:, 0, :]
        context_cls = context_out[:, 0, :]
        if use_cross_attention:
            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=768)
            query = tf.expand_dims(context_cls, axis=1)
            attn_output = mha(query, text_out, text_out)
            combined = tf.squeeze(attn_output, axis=1)
        else:
            combined = tf.concat([text_cls, context_cls], axis=-1)
    else:
        pos_emb_text = PositionalEmbedding(text_maxlen, vocab_size, embedding_dim)
        pos_emb_context = PositionalEmbedding(ctx_maxlen, vocab_size, embedding_dim)
        encoder_text = TransformerEncoder(num_layers, embedding_dim, num_heads, dff, dropout)
        encoder_context = TransformerEncoder(num_layers, embedding_dim, num_heads, dff, dropout)
        text_emb = pos_emb_text(text_input)
        context_emb = pos_emb_context(context_input)
        text_out = encoder_text(text_emb)
        context_out = encoder_context(context_emb)
        text_pooled = tf.reduce_mean(text_out, axis=1)
        context_pooled = tf.reduce_mean(context_out, axis=1)
        if use_cross_attention:
            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
            attn_output = mha(context_out, text_out, text_out)
            combined = tf.reduce_mean(attn_output, axis=1)
        else:
            combined = tf.concat([text_pooled, context_pooled], axis=-1)

    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[text_input, context_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model