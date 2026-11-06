from chatbot import _append_norm_warning


def test_warns_about_article_outside_context():
    reply = "Применима ст. 178 ТК РФ."
    result = _append_norm_warning(reply, [{"snippet": "Суд применил ст. 394 ТК РФ."}])
    assert "Примечание" in result
    assert "178" in result


def test_does_not_warn_when_article_is_in_context():
    reply = "Применима ст. 394 ТК РФ."
    result = _append_norm_warning(reply, [{"snippet": "Суд применил ст. 394 ТК РФ."}])
    assert "Примечание" not in result


def test_filename_is_not_presented_as_case_number():
    reply = "Дело case_1.txt рассмотрено судом."
    result = _append_norm_warning(reply, [{"filename": "case_1.txt", "snippet": "текст"}])
    assert "Документ case_1.txt" in result
    assert "Дело case_1.txt" not in result


def test_filename_without_extension_is_not_presented_as_case_number():
    reply = "В деле № abcdefgh12 суд восстановил работника."
    result = _append_norm_warning(
        reply, [{"filename": "abcdefgh12.txt", "snippet": "текст"}]
    )
    assert "документе abcdefgh12.txt" in result
    assert "деле № abcdefgh12" not in result
