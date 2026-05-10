from backend.chunking.token_splitter import TokenSplitter


def test_token_splitter_counts_tokens_and_respects_hard_max() -> None:
    splitter = TokenSplitter(ideal_max=80, hard_max=120, overlap=20)
    text = " ".join([f"Sentence {index} has useful admissions context." for index in range(120)])

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(splitter.count_tokens(chunk) <= 120 for chunk in chunks)


def test_token_splitter_keeps_bullet_block_together_when_possible() -> None:
    splitter = TokenSplitter(hard_max=1000)
    text = "- Eligibility: JEE Main\n- Documents: Marksheet\n- Fees: Published by admissions office"

    assert splitter.split_text(text) == [text]

