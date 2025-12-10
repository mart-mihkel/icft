import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    import logging

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from cptlms.bert import PTuningBert
    from cptlms.squad import Squad

    logging.basicConfig(level="INFO")
    torch.set_float32_matmul_precision("high")


@app.cell
def _():
    _pretrained_model = "bert-base-uncased"
    _tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    _base_bert = AutoModelForQuestionAnswering.from_pretrained(_pretrained_model)

    pt_bert = PTuningBert(_base_bert, num_virtual_tokens=128)
    squad = Squad(_tokenizer)
    return pt_bert, squad


@app.cell
def _(squad):
    _x = squad.train_tok.with_format("pt")[0:1]
    input_ids = _x["input_ids"]
    attention_mask = _x["attention_mask"]
    start_positions = _x["start_positions"]
    end_positions = _x["end_positions"]
    batch_size = input_ids.size(0)
    return (
        attention_mask,
        batch_size,
        end_positions,
        input_ids,
        start_positions,
    )


@app.cell
def _(attention_mask, batch_size, input_ids, pt_bert):
    # batch, virtual, hidden
    virtual_embeds = pt_bert.prompt_encoder().unsqueeze(0).expand(batch_size, -1, -1)

    # batch, sequence, hidden
    bert_input_embeds = pt_bert.bert_embedding(input_ids)

    # batch, virtual
    virtual_attention = torch.ones(
        batch_size,
        pt_bert.num_virtual_tokens,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return bert_input_embeds, virtual_attention, virtual_embeds


@app.cell
def _(
    attention_mask,
    bert_input_embeds,
    end_positions,
    pt_bert,
    start_positions,
    virtual_attention,
    virtual_embeds,
):
    pt_bert.bert(
        inputs_embeds=torch.cat([virtual_embeds, bert_input_embeds], dim=1),
        attention_mask=torch.cat([virtual_attention, attention_mask], dim=1),
        start_positions=start_positions,
        end_positions=end_positions,
    )
    return


if __name__ == "__main__":
    app.run()
