import tensorflow as tf
import numpy as np

class ResidualNormalizationWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, dropout_rate, *args, **kwargs):
        super().__init__(layer, *args, **kwargs)
        self.dropout_rate = dropout_rate
        self.layer_normalization = tf.keras.layers.LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, *args, training = False, **kwargs):
        x = self.layer_normalization(inputs)
        x = self.layer(x, *args, training=training, **kwargs)
        x = self.dropout_layer(x, training=training)
        return inputs + x

    def get_config(self):
        config = {
            "dropout_rate": self.dropout_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AddPositionalEncoding(tf.keras.layers.Layer):
    '''
    入力テンソルに対し、位置の情報を付与して返すレイヤーです。
    see: https://arxiv.org/pdf/1706.03762.pdf

    PE_{pos, 2i}   = sin(pos / 10000^{2i / d_model})
    PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_model})
    '''
    def get_angles(self, pos, i, d_model):
        angle_rates = 1. / tf.math.pow(10000., tf.cast(2 * (i//2), dtype=tf.float32) / tf.cast(d_model, dtype=tf.float32))
        return tf.cast(pos, dtype=tf.float32) * angle_rates

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        fl_type = inputs.dtype
        batch_size, max_length, depth = tf.unstack(tf.shape(inputs))

        angle_rads = self.get_angles(tf.range(max_length)[:, tf.newaxis],
                          tf.range(depth)[tf.newaxis, :],
                          depth)

        pos_encoding = tf.stack([tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [max_length, depth])[tf.newaxis,...]

        return inputs + tf.cast(pos_encoding, dtype=fl_type)

class FeedForwardNetwork(tf.keras.models.Model):
    '''
    Transformer 用の Position-wise Feedforward Neural Network です。
    '''
    def __init__(self, hidden_dim: int, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4, use_bias=True,
                                                        activation='relu', name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=True, name='output_layer')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input, training = False):
        '''
        FeedForwardNetwork を適用します。
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        x = self.filter_dense_layer(input)
        x = self.dropout_layer(x, training=training)
        x = self.output_dense_layer(x)
        return x

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class Encoder(tf.keras.models.Model):
    def __init__(
            self,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.input_dense = tf.keras.layers.Dense(hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            attention_layer = tf.keras.layers.MultiHeadAttention(head_num, hidden_dim, dropout=dropout_rate)
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate)
            self.attention_block_list.append([
                ResidualNormalizationWrapper(attention_layer, dropout_rate),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate),
            ])
        self.output_normalization = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training = False):
        '''
        モデルを実行します

        :param input: shape = [batch_size, length, hidden_dim]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        # [batch_size, length, hidden_dim]
        embedded_input = self.input_dense(inputs)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        mask = tf.math.reduce_any(inputs != 0, axis=-1)
        self_attention_mask = tf.logical_and(mask[...,tf.newaxis,:],mask[...,:,tf.newaxis])

        for i, layers in enumerate(self.attention_block_list):
            attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = attention_layer(query, query, attention_mask=self_attention_mask, training=training)
                query = ffn_layer(query, training=training)
        # [batch_size, length, hidden_dim]
        return self.output_normalization(query)

    def get_config(self):
        return {
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class Decoder(tf.keras.models.Model):
    '''
    エンコードされたベクトル列からトークン列を生成する Decoder です。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num: int,
            head_num: int,
            hidden_dim: int,
            dropout_rate: float,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num = hopping_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.embedding_layer = tf.keras.layers.Embedding(vocab_size + 2, hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.input_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.attention_block_list: List[List[tf.keras.models.Model]] = []
        for _ in range(hopping_num):
            self_attention_layer = tf.keras.layers.MultiHeadAttention(head_num, hidden_dim, dropout=dropout_rate)
            enc_dec_attention_layer = tf.keras.layers.MultiHeadAttention(head_num, hidden_dim, dropout=dropout_rate)
            ffn_layer = FeedForwardNetwork(hidden_dim, dropout_rate)
            self.attention_block_list.append([
                ResidualNormalizationWrapper(self_attention_layer, dropout_rate),
                ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate),
                ResidualNormalizationWrapper(ffn_layer, dropout_rate),
            ])
        self.output_normalization = tf.keras.layers.LayerNormalization()
        self.output_dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training = False):
        '''
        モデルを実行します

        :param input: shape = [batch_size, length]
        :param training: 学習時は True
        :return: shape = [batch_size, length, hidden_dim]
        '''
        decoder_input, encoder_output, encoder_input = inputs

        # [batch_size, length, hidden_dim]
        embedded_input = self.embedding_layer(decoder_input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.input_dropout_layer(embedded_input, training=training)

        mask_decoder = decoder_input != 0
        mask_encoder = tf.math.reduce_any(encoder_input != 0, axis=-1)

        self_attention_mask = tf.logical_and(mask_decoder[...,tf.newaxis,:],mask_decoder[...,:,tf.newaxis])

        enc_dec_attention_mask = tf.logical_and(mask_decoder[...,:,tf.newaxis],mask_encoder[...,tf.newaxis,:])

        for i, layers in enumerate(self.attention_block_list):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)
            with tf.name_scope(f'hopping_{i}'):
                query = self_attention_layer(query, query, attention_mask=self_attention_mask, training=training)
                query = enc_dec_attention_layer(query, encoder_output,
                                                attention_mask=enc_dec_attention_mask, training=training)
                query = ffn_layer(query, training=training)

        query = self.output_normalization(query)  # [batch_size, length, hidden_dim]
        return self.output_dense(query)  # [batch_size, length, vocab_size]

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }

class TextTransformer(tf.keras.models.Model):
    '''
    Transformer モデルです。
    '''
    def __init__(
            self,
            vocab_size: int,
            hopping_num_encoder: int = 4,
            hopping_num_decoder: int = 4,
            head_num: int = 8,
            hidden_dim: int = 512,
            dropout_rate: float = 0.1,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hopping_num_encoder = hopping_num_encoder
        self.hopping_num_decoder = hopping_num_decoder
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(
            hopping_num=hopping_num_encoder,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hopping_num=hopping_num_decoder,
            head_num=head_num,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def call(self, inputs, training = False):
        encoder_input, decoder_input = inputs
        encoder_output = self.encoder(
            encoder_input,
            training=training,
        )
        decoder_output = self.decoder(
            (decoder_input, encoder_output, encoder_input),
            training=training,
        )
        return decoder_output

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hopping_num": self.hopping_num,
            "head_num": self.head_num,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        }


if __name__ == '__main__':
    transformer = TextTransformer(vocab_size=256)
    embedded = tf.keras.Input(shape=(None,256))
    decoderinput = tf.keras.Input(shape=(None,))
    outputs = transformer((embedded, decoderinput))
    model = tf.keras.Model(inputs={'encoder': embedded, 'decoder': decoderinput}, outputs=outputs)
    model.summary()
