from plainline.readability import fk_grade, in_band, smog_grade, syllables, word_count


def test_syllables():
    assert syllables("cat") == 1
    assert syllables("medicine") == 3
    assert syllables("anticoagulant") >= 5


def test_grade_orders_texts():
    plain = "Take one pill each day. Do not take more. Call your doctor if you bleed."
    dense = ("Concomitant administration of anticoagulants necessitates meticulous "
             "international normalized ratio surveillance throughout therapeutic initiation.")
    assert fk_grade(plain) < 8 < fk_grade(dense)
    assert smog_grade(plain) < smog_grade(dense)


def test_empty_and_band():
    assert fk_grade("") == 0.0
    assert word_count("a b c") == 3
    assert in_band(5, None) is None
    assert in_band(5, (None, 8)) is True
    assert in_band(9, (None, 8)) is False
    assert in_band(9, (10, None)) is False


def test_spearman_handles_ties():
    from plainline.metrics import length_control

    rows = [{"cfr": c, "words": w} for c, w in [(1.0, 10), (1.0, 20), (1.0, 30), (0.5, 40), (0.5, 50), (0.5, 60)]]
    rho = length_control(rows)["spearman_cfr_vs_words"]
    assert abs(rho - (-0.8783101)) < 1e-4
