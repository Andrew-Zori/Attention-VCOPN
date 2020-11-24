# Attention-VCOPN
Modified VCOPN model for COMP4471 Project

## Highlight in the modification for VCOPN class
Go to the model and click the vcopn.py file. Inside the file, there should be two classes, named VCOPN and VCOPN_attention respectively.   

VCOPN is the original class from the paper and VCOPN_attention is the modified one.
  
For difference is in the forward definition after base_network:
```python
# Original Paper
        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                # torch.cat -> (Batch, Feature_size1 + Feature_size2)
                # concat the output horizontally, instead of vertically shown in the paper
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]

        # Size -> (Batch, Tuple_len * Feature_size)
        h = torch.cat(pf, dim = 1)

# Modified part
        # (Tuple_len, Batch, Feature_size) -> (Batch, Tuple_len, Feature_size)
        encoder_inp = torch.stack(f).permute(1,0,2)
        
        # Size Still (Batch, Tuple_len, Feature_size)
        encoder_out = self.encoder(encoder_inp)

        pf = [encoder_out[:,i,:] for i in range(encoder_out.size(1))]

        # Size -> (Batch, Tuple_len * Feature_size)
        h = torch.cat(pf, dim = 1)
```
The modified part utilized an attention encoder to return a (batch_size, tuple_length, feature_size =512) output.

## Initialization of attention_vcopn model object

You may go to the Attention_VCOPN.ipynb under the model folder to check the example of how the model object runs.

The initialization code is:

```python
# Generator, a simple nn.Linear layer. 
# It's a dummy part for this case. I didn't change it to avoid potential bugs.
generator = Generator(512, 4)

# initialize encoder
encoder = make_encoder(generator, h = 4)

vcopn = VCOPN_attention(base_network=base, feature_size=512, tuple_len=3, encoder = encoder)
```

### make_encoder function:

The make_encoder function is also defined in the attention_vcopn.ipynb file. It is written as:

```python
"""
import copy
deepcopy = copy.deepcopy
"""
    # h for heads in the attention layer, 
    # d_model for input feature_size and output feature size 
    attn = MultiHeadAttention(h, d_model)

    # d_model for input feature size and output feature size
    # ff structure as nn.Linear (d_model, d_ff) x (d_ff, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # EncoderLayer as one layer of encoder,
    # N for the number of encoder layers within the encoder
    # Here I suppose N = 1 should be sufficient
    encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), 
    deepcopy(ff), dropout), generator, N)
```
