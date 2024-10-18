## mamba-tiny

Tiny implementation of Mamba in PyTorch.

Featuring:
* Equivalent numerical output as official implementation for both forward and backward pass
* Simplified, readable, annotated code
* An alternative to using parallel scan (not available in pytorch as of current) via cumsum, 
inspired by [heisen_sequence](https://github.com/glassroom/heinsen_sequence)

Does NOT include:
* Recurrent mode of the network intended for inference. The demo code (sentence generation) effectively runs the network as if it were the forward pass during training, which is much slower than the recurrent mode.
* Kernel fusion. This repo does not make any attempt to perform kernel fusion of the selective scan operations with the other dense operations. So all the internal states of the model would be explicitly materialized, so memory usage may be a bottleneck.
* Proper parameter initialization (though this could be added without sacrificing readability)

## Demo

See [demo.ipynb](demo.ipynb) for examples of prompt completions.

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba is the world's longest venomous snake with an estimated length of over 150 m. With such a large size and a venomous bite, Mamba kills by stabbing the victim (which is more painful and less effective than a single stab of the bite)

150 meters... 🫢 scary!

## References

The [Mamba](https://arxiv.org/abs/2312.00752) architecture was introduced by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor). The official implementation is here: https://github.com/state-spaces/mamba/tree/main

Related works using parallel scans in log-space:
* [miniGRU and miniLSTM](https://arxiv.org/abs/2410.01201)
* [FlashLinearAttention](https://github.com/sustcsonglin/flash-linear-attention)
