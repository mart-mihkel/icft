from typing import cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from datasets.utils.info_utils import VerificationMode
from peft import PeftModel
from pytest import fixture
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from instruct.modeling import get_pt_model
from instruct.types import DatasetInfo

_split = {
    "train": "train[:10]",
    "validation": "validation[:10]",
    "test": "test[:10]",
}

_split_estner = {
    "train": "train[:10]",
    "dev": "dev[:10]",
    "test": "test[:10]",
}

_info = DatasetInfo(
    id2label={0: "0", 1: "1"},
    label2id={"0": 0, "1": 1},
    system_prompt="test",
)

_bert = "hf-internal-testing/tiny-random-bert"
_gpt2 = "hf-internal-testing/tiny-random-gpt2"
_t5 = "hf-internal-testing/tiny-random-t5"
_llama = "hf-internal-testing/tiny-random-llama4"
_gemma = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
_qwen = "Jiqing/tiny-random-qwen2"


@fixture(scope="session")
def bert() -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(_bert)


@fixture(scope="session")
def pt_bert(bert_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", bert_tokenizer, _bert, "encoder", _info)


@fixture(scope="session")
def bert_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_bert)
    return cast(PreTrainedTokenizerFast, tokenizer)


@fixture(scope="session")
def gpt2() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(_gpt2)
    return cast(PreTrainedModel, model)


@fixture(scope="session")
def pt_gpt2(gpt2_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", gpt2_tokenizer, _gpt2, "decoder", _info)


@fixture(scope="session")
def gpt2_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_gpt2)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def t5() -> PreTrainedModel:
    model = AutoModelForSeq2SeqLM.from_pretrained(_t5)
    return cast(PreTrainedModel, model)


@fixture(scope="session")
def pt_t5(t5_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", t5_tokenizer, _t5, "encoder-decoder", _info)


@fixture(scope="session")
def t5_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_t5)
    return cast(PreTrainedTokenizerFast, tokenizer)


@fixture(scope="session")
def llama() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(_llama)
    return cast(PreTrainedModel, model)


@fixture(scope="session")
def pt_llama(llama_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", llama_tokenizer, _llama, "decoder", _info)


@fixture(scope="session")
def llama_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_llama)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def gemma() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(_gemma)
    return cast(PreTrainedModel, model)


@fixture(scope="session")
def pt_gemma(gemma_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", gemma_tokenizer, _gemma, "decoder", _info)


@fixture(scope="session")
def gemma_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_gemma)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def qwen() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(_qwen)
    return cast(PreTrainedModel, model)


@fixture(scope="session")
def pt_qwen(qwen_tokenizer: PreTrainedTokenizerFast) -> PeftModel:
    return get_pt_model("random", qwen_tokenizer, _qwen, "decoder", _info)


@fixture(scope="session")
def qwen_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(_qwen)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@fixture(scope="session")
def boolq() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset("super_glue", "boolq", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def wic() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset("super_glue", "wic", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def estner() -> DatasetDict:
    split = cast(Split, _split_estner)
    data = load_dataset("tartuNLP/EstNER", split=split)
    return cast(DatasetDict, data)


@fixture(scope="session")
def multinerd() -> DatasetDict:
    split = cast(Split, _split)
    data = load_dataset(
        "Babelscape/multinerd",
        verification_mode=VerificationMode.NO_CHECKS,
        split=split,
    )

    return cast(DatasetDict, data)
