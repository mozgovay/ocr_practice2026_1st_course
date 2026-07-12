"""
MVP сайта: загружаешь скан рукописного ответа -> получаешь распознанный текст.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from trocr import (
    find_text_lines,
    split_line_into_chunks,
    recognize_line,
    group_lines_into_tasks,
)

st.set_page_config(page_title="Распознавание ответов ЕГЭ")

st.title("📝 Распознавание рукописных ответов ЕГЭ")
st.write(
    "Загрузи скан страницы с развёрнутым ответом — сервис распознает текст "
    "и попробует разбить его по номерам заданий."
)

uploaded_file = st.file_uploader("Скан ответа", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Оригинал", use_container_width=True)

    # улучшенная версия для предпросмотра (контраст + резкость)
    preview_gray = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    preview_enhanced = clahe.apply(preview_gray)
    preview_enhanced = cv2.convertScaleAbs(preview_enhanced, alpha=1.5, beta=10)

    with col2:
        st.image(preview_enhanced, caption="Улучшенное", use_container_width=True, clamp=True)

    if st.button("Распознать текст"):
        gray = np.array(image.convert("L"))

        with st.spinner("Подготовка изображения..."):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=15)
            gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

        with st.spinner("Ищу строки текста..."):
            lines = find_text_lines(gray)

        if not lines:
            st.warning("Строки не найдены. Попробуйте другое изображение.")
            st.stop()

        st.write(f"Найдено строк: **{len(lines)}**")

        progress_bar = st.progress(0)
        status_text = st.empty()

        line_texts = []
        for i, (y1, y2) in enumerate(lines):
            status_text.text(f"Распознаю строку {i + 1} из {len(lines)}...")

            y1 = max(0, y1 - 3)
            y2 = min(gray.shape[0], y2 + 3)
            line_gray = gray[y1:y2, :]

            # увеличение разрешения
            h, w = line_gray.shape
            if 0 < h < 25:
                scale = 40 / h
                new_w = int(w * scale)
                line_gray = cv2.resize(line_gray, (new_w, 40), interpolation=cv2.INTER_CUBIC)

            chunks = split_line_into_chunks(line_gray)

            chunk_texts = []
            for x1, x2 in chunks:
                x1 = max(0, x1 - 5)
                x2 = min(line_gray.shape[1], x2 + 5)
                crop = line_gray[:, x1:x2]

                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 10:
                    text = recognize_line(crop)
                    if text and len(text.strip()) > 0:
                        chunk_texts.append(text.strip())

            line_text = " ".join(chunk_texts)
            line_texts.append(line_text)
            progress_bar.progress((i + 1) / len(lines))

        status_text.text("✅ Готово!")
        progress_bar.empty()

        full_text = "\n".join(line_texts)

        st.subheader("📄 Полный распознанный текст")
        st.text_area("", full_text, height=300)

        st.subheader("📌 Разбивка по заданиям")
        blocks = group_lines_into_tasks(line_texts, lines)

        if blocks:
            for task_num in sorted(blocks.keys()):
                if task_num != 0:
                    with st.expander(f"Задание {task_num}"):
                        st.write(blocks[task_num])
        else:
            st.info("Не удалось автоматически разбить текст по заданиям.")
            with st.expander("📝 Показать все строки"):
                for i, text in enumerate(line_texts):
                    if text.strip():
                        st.write(f"{i + 1}. {text}")

        st.download_button(
            label="💾 Скачать текст",
            data=full_text,
            file_name="recognized_text.txt",
            mime="text/plain",
        )