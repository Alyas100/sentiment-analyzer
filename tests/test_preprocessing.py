from sentiment_analyzer.preprocessing import detect_language, expand_slang, preprocess


def test_preprocess_expands_slang_and_cleans():
    out = preprocess("Barang sampai cepat giler!!! 😍 https://shop.example/x")
    assert "sangat" in out
    assert "http" not in out
    assert "😍" not in out
    assert "!!!" not in out


def test_expand_slang_word_boundaries():
    assert "sangat" in expand_slang("giler")
    assert "sangat" in expand_slang("GILER")
    # should not replace inside another word
    assert expand_slang("gileran") == "gileran"


def test_detect_language_malay_english():
    assert detect_language("This product is good") == "en"
    assert detect_language("barang ni tak best lah") in ("ms-en", "ms")


def test_detect_language_pure_malay_latin_script():
    # ASCII-only Malay must not be labelled English
    assert detect_language("Kualiti barang bagus, harga juga murah") == "ms"
    assert detect_language("Barang sampai cepat giler. Seller mantap!") == "ms"
    assert detect_language("Baju nipis sangat tak sama macam dalam gambar.") == "ms"


def test_detect_language_code_mixed():
    assert (
        detect_language("Customer service teruk chat tak reply langsung! Nak refund.") == "ms-en"
    )


def test_detect_language_english_only():
    assert detect_language("Very fast delivery, seller is great!") == "en"
    assert detect_language("The packaging was damaged and the item arrived broken") == "en"


def test_detect_language_unknown():
    assert detect_language("") == "unknown"
    assert detect_language("12345 !!!") == "unknown"


def test_preprocess_empty():
    assert preprocess("") == ""
