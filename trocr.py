"""
Тест: распознавание рукописного текста через TrOCR (модель дообучена под рукописную
кириллицу: kazars24/trocr-base-handwritten-ru

Подход:
1. Бинаризуем скан (чёрно-белый, текст = тёмный на светлом)
2. Находим горизонтальные полосы текста через проекцию (сумма тёмных пикселей по строкам)
3. Вырезаем каждую строку как отдельное изображение
4. Прогоняем каждую строку через TrOCR
5. Склеиваем в полный текст, разбиваем по номерам заданий
"""

import re

import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = "kazars24/trocr-base-handwritten-ru"

_processor = None
_model = None


def get_model():
    global _processor, _model
    if _model is None:
        _processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    return _processor, _model


def load_image_as_array(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")  # grayscale
    return np.array(img)


def _find_text_lines_raw(gray: np.ndarray, min_line_height: int, padding: int,
                          threshold_pct: float) -> list[tuple[int, int]]:
    """Один проход горизонтальной проекции с заданным порогом плотности текста."""
    threshold = gray.mean() * 0.85
    dark_mask = gray < threshold

    row_sums = dark_mask.sum(axis=1)
    has_text = row_sums > (gray.shape[1] * threshold_pct)

    lines = []
    in_line = False
    start = 0
    for y, val in enumerate(has_text):
        if val and not in_line:
            start = y
            in_line = True
        elif not val and in_line:
            end = y
            if end - start >= min_line_height:
                lines.append((max(0, start - padding), min(gray.shape[0], end + padding)))
            in_line = False
    if in_line:
        lines.append((max(0, start - padding), gray.shape[0]))
    return lines


def find_text_lines(gray: np.ndarray, min_line_height: int = 15, padding: int = 8,
                     threshold_pct: float = 0.04, _depth: int = 0) -> list[tuple[int, int]]:
    """
    Нахождение вертикальных диапазонов (y_start, y_end), где есть текст,
    через горизонтальную проекцию тёмных пикселей.
    """
    lines = _find_text_lines_raw(gray, min_line_height, padding, threshold_pct)
    if not lines or _depth >= 2:
        return lines

    heights = sorted(y2 - y1 for y1, y2 in lines)
    median_h = heights[len(heights) // 2]

    result = []
    for (y1, y2) in lines:
        h = y2 - y1
        if h > median_h * 1.7 and threshold_pct < 0.1:
            sub_gray = gray[y1:y2, :]
            sub_lines = find_text_lines(sub_gray, min_line_height, padding,
                                         threshold_pct + 0.02, _depth + 1)
            if len(sub_lines) > 1:
                result.extend((y1 + a, y1 + b) for a, b in sub_lines)
                continue
        result.append((y1, y2))

    return [(a, b) for a, b in result if b - a >= min_line_height]


def split_line_into_chunks(line_gray: np.ndarray, max_chunk_width: int = 700,
                            min_gap_width: int = 6) -> list[tuple[int, int]]:
    threshold = line_gray.mean() * 0.85
    dark_mask = line_gray < threshold
    col_sums = dark_mask.sum(axis=0)
    is_gap = col_sums == 0

    # находим непрерывные пустые промежутки >= min_gap_width — это границы слов
    gaps = []
    in_gap = False
    start = 0
    for x, v in enumerate(is_gap):
        if v and not in_gap:
            start = x
            in_gap = True
        elif not v and in_gap:
            if x - start >= min_gap_width:
                gaps.append((start, x))
            in_gap = False

    # границы кусков — середины промежутков между словами
    boundaries = [0] + [(g[0] + g[1]) // 2 for g in gaps] + [line_gray.shape[1]]

    # группировка слов в куски шириной не больше max_chunk_width
    chunks = []
    chunk_start = boundaries[0]
    for i in range(1, len(boundaries)):
        if boundaries[i] - chunk_start > max_chunk_width and boundaries[i - 1] > chunk_start:
            chunks.append((chunk_start, boundaries[i - 1]))
            chunk_start = boundaries[i - 1]
    chunks.append((chunk_start, boundaries[-1]))

    return [(a, b) for a, b in chunks if b > a]


def recognize_line(image_crop: np.ndarray) -> str:
    processor, model = get_model()
    pil_img = Image.fromarray(image_crop).convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=32, num_beams=1)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def ocr_scan_trocr(image_path: str) -> str:
    gray = load_image_as_array(image_path)
    lines = find_text_lines(gray)
    print(f"Найдено строк: {len(lines)}")

    texts = []
    for i, (y1, y2) in enumerate(lines):
        line_gray = gray[y1:y2, :]
        chunks = split_line_into_chunks(line_gray)

        chunk_texts = []
        for x1, x2 in chunks:
            crop = line_gray[:, x1:x2]
            text = recognize_line(crop)
            chunk_texts.append(text)

        line_text = " ".join(chunk_texts)
        texts.append(line_text)

    return "\n".join(texts)


def find_task_boundaries(lines: list[tuple[int, int]], gap_threshold: int = 20,
                          cluster_window: int = 3) -> list[int]:
    """
    Находдение индексов строк, с которых начинается новое задание, по промежуткам
    между абзацами (расстояние между концом предыдущей строки и началом следующей).
    """
    boundaries = [0]
    last_boundary = -(cluster_window + 1)
    for i in range(1, len(lines)):
        gap = lines[i][0] - lines[i - 1][1]
        if gap > gap_threshold and (i - last_boundary) > cluster_window:
            boundaries.append(i)
            last_boundary = i
    return boundaries


TASK_NUM_HINT_PATTERN = re.compile(r"^\s*(\d{1,2})\s*[.,)]")


def group_lines_into_tasks(line_texts: list[str], lines: list[tuple[int, int]]) -> dict[int, str]:
    """
    Группировка строки в блоки по заданиям, используя find_task_boundaries.
    """
    boundaries = find_task_boundaries(lines)
    blocks: dict[int, str] = {}

    prev_task_num = None
    for i, start_idx in enumerate(boundaries):
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(line_texts)
        block_lines = line_texts[start_idx:end_idx]
        block_text = " ".join(block_lines).strip()

        match = TASK_NUM_HINT_PATTERN.match(block_text)
        if match and 15 <= int(match.group(1)) <= 35:
            task_num = int(match.group(1))
        elif prev_task_num is not None:
            task_num = prev_task_num + 1
        else:
            task_num = 0

        blocks[task_num] = block_text
        if task_num != 0:
            prev_task_num = task_num

    return blocks


TASK_NUM_PATTERN = re.compile(r"^\s*(\d{1,2})\s*[.,)]\s*")


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/0319535382_02.png"

    print(f"Распознаю: {image_path}")
    full_text = ocr_scan_trocr(image_path)

    print("\n--- Полный текст ---")
    print(full_text)

    print("\n--- Разбивка по заданиям ---")
    blocks = group_lines_into_tasks(full_text)
    for task_num, text in blocks.items():
        preview = text[:150] + ("..." if len(text) > 150 else "")
        print(f"\n[{task_num}] {preview}")
