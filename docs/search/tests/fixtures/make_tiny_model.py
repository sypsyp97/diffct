"""Write a tiny stand-in for the search embedding model, so that the browser tests run the real
transformers.js runtime without the 377 MB download.

The model maps every character token t to sin(t * f) over 512 fixed frequencies f, masked by the
attention mask. Mean pooling of these rows gives a deterministic vector that differs per query.

Run: uv run --no-project --with onnx==1.19.1 --with numpy python docs/search/tests/fixtures/make_tiny_model.py
"""
import json
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

HERE = Path(__file__).parent / "tiny-model"
DIM = 512

freqs = np.array([0.37 + 0.91 * math.sin(1.7 * i) ** 2 for i in range(DIM)], dtype=np.float32)
nodes = [
    helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["ids_f", "axis"], ["ids_3d"]),
    helper.make_node("Mul", ["ids_3d", "freqs"], ["phase"]),
    helper.make_node("Sin", ["phase"], ["wave"]),
    helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["mask_f", "axis"], ["mask_3d"]),
    helper.make_node("Mul", ["wave", "mask_3d"], ["last_hidden_state"]),
]
graph = helper.make_graph(
    nodes,
    "tiny_embedder",
    [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
    ],
    [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", DIM])],
    initializer=[
        numpy_helper.from_array(freqs, "freqs"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "axis"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="diffct-tests")
model.ir_version = 8
onnx.checker.check_model(model)
(HERE / "onnx").mkdir(parents=True, exist_ok=True)
# Like the real checkpoint, the weights live in an external data file next to the graph.
for old in (HERE / "onnx").glob("model_quantized.onnx*"):
    old.unlink()
onnx.save(model, HERE / "onnx" / "model_quantized.onnx", save_as_external_data=True,
          all_tensors_to_one_file=True, location="model_quantized.onnx_data", size_threshold=1024)

# One token per printable ASCII character; everything else maps to [UNK].
vocab = {"[UNK]": 0, "[PAD]": 1}
for code in range(33, 127):
    vocab[chr(code)] = len(vocab)
tokenizer = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 0, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True},
        {"id": 1, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True},
    ],
    "normalizer": {"type": "Lowercase"},
    "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": False},
    "post_processor": None,
    "decoder": None,
    "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
}
(HERE / "tokenizer.json").write_text(json.dumps(tokenizer, indent=1) + "\n")
(HERE / "tokenizer_config.json").write_text(json.dumps({
    "tokenizer_class": "PreTrainedTokenizer",
    "model_max_length": 512,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
}, indent=1) + "\n")
(HERE / "config.json").write_text(json.dumps({
    "model_type": "bert",
    "architectures": ["BertModel"],
    "hidden_size": DIM,
    "num_attention_heads": 1,
    "num_hidden_layers": 1,
    "vocab_size": len(vocab),
    "transformers.js_config": {"use_external_data_format": {"model_quantized.onnx": 1}},
}, indent=1) + "\n")
print("wrote", sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()))
