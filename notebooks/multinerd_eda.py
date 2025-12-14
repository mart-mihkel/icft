import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    from datasets import load_dataset, VerificationMode
    from transformers import AutoTokenizer


@app.cell
def _():
    tag2id = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    id2tag = {v: k for k, v in tag2id.items()}
    id2tag
    return


@app.cell
def _():
    dataset = load_dataset(
        "Babelscape/multinerd",
        verification_mode=VerificationMode.NO_CHECKS,
    )

    dataset
    return (dataset,)


@app.cell
def _(dataset):
    def _en_filter(batch: dict[str, list]) -> list[bool]:
        return [lang == "en" for lang in batch["lang"]]

    train = dataset["train"].filter(_en_filter, batched=True)
    # val = dataset["validation"].filter(_en_filter, batched=True)
    # test = dataset["test"].filter(_en_filter, batched=True)

    train
    return (train,)


@app.cell
def _(train):
    batch = train[0:5]
    batch
    return


@app.cell
def _():
    pretrained_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer
    return (tokenizer,)


@app.cell
def _(tokenizer, train):
    def _align_words_to_tags(word_ids, ner_tags):
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(ner_tags[word_id])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        return label_ids

    def _tokenize(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, ner_tags in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            tag_ids = _align_words_to_tags(word_ids=word_ids, ner_tags=ner_tags)
            labels.append(tag_ids)

        tokenized["labels"] = labels
        return tokenized

    train_tokenized = train.map(_tokenize, batched=True)
    train_tokenized
    return


@app.cell
def _(tokenizer, train):
    sep_token = tokenizer.special_tokens_map["sep_token"]
    assert isinstance(sep_token, str), "Expected one separator token"

    cls_token = tokenizer.special_tokens_map["cls_token"]
    assert isinstance(cls_token, str), "Expected one classification token"

    system_prompt = (
        "Task : Determine the named entity tag . Question: What is the NER tag of "
        "<word> in the sentence <sentence> ? Possible tags : O PER ORG LOC.".split()
    )

    example = train[0]
    tokens = example["tokens"]
    target_token = tokens[4]

    (
        [cls_token]
        + system_prompt
        + [sep_token]
        + ["<word>", target_token, "<sentence>"]
        + tokens
        + [sep_token]
    )
    return


if __name__ == "__main__":
    app.run()
