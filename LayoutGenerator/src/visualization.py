import os

import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont

from .utilities import CANVAS_SIZE, ID2LABEL, RAW_DATA_PATH


class Visualizer:
    def __init__(self, dataset: str, times: float = 3):
        self.dataset = dataset
        self.times = times
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]
        self._colors = None

    def draw_layout_with_content(self, labels: torch.Tensor, bboxes: torch.Tensor, content_data: dict, show_bbox: bool = True):
        """
        레이아웃을 그리면서 바운딩 박스 안에 해당하는 텍스트 내용을 추가합니다.
        
        Args:
            labels: 레이블 텐서
            bboxes: 바운딩 박스 텐서
            content_data: {'title': '제목', 'description': '설명', 'image': '이미지 설명', 'button': '버튼 텍스트'}
            show_bbox: 바운딩 박스를 보여줄지 여부 (기본값: True)
        """
        _canvas_width = self.canvas_width * self.times
        _canvas_height = self.canvas_height * self.times
        img = Image.new("RGB", (int(_canvas_width), int(_canvas_height)), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        
        # 기본 폰트 설정 (시스템에 따라 달라질 수 있음)
        try:
            # macOS의 기본 한글 폰트
            font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
                small_font = ImageFont.truetype(font_path, 14)
            else:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        labels_list = labels.tolist()
        bboxes_list = bboxes.tolist()
        areas = [bbox[2] * bbox[3] for bbox in bboxes_list]
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        # ID2LABEL 매핑 생성 (숫자 -> 라벨명)
        id_to_label = ID2LABEL[self.dataset]

        for i in indices:
            bbox, label = bboxes_list[i], labels_list[i]
            color = self.colors[label]
            c_fill = color + (100,)
            x1, y1, x2, y2 = bbox
            x2 += x1
            y2 += y1
            x1, x2 = x1 * _canvas_width, x2 * _canvas_width
            y1, y2 = y1 * _canvas_height, y2 * _canvas_height
            
            # 바운딩 박스 그리기 (옵션에 따라)
            if show_bbox:
                draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
            
            # 레이블명 가져오기
            label_name = id_to_label.get(label, f"label_{label}")
            
            # 해당 레이블에 맞는 텍스트 내용 가져오기
            text_content = content_data.get(label_name, "")
            
            if text_content:
                # 텍스트 영역 계산
                text_x = x1 + 5
                text_y = y1 + 5
                max_width = x2 - x1 - 10
                max_height = y2 - y1 - 10
                
                # 텍스트가 박스 안에 들어가도록 조정
                if label_name in ['title', 'button']:
                    # 제목이나 버튼은 한 줄로
                    self._draw_text_in_box(draw, text_content, text_x, text_y, max_width, max_height, font, (0, 0, 0))
                else:
                    # 설명이나 이미지 설명은 여러 줄 가능
                    self._draw_multiline_text_in_box(draw, text_content, text_x, text_y, max_width, max_height, small_font, (0, 0, 0))
        
        return img
    
    def _find_optimal_font_size(self, draw, text, max_width, max_height, single_line=True, min_size=8, max_size=100):
        """주어진 박스 크기에 맞는 최적의 폰트 크기를 찾습니다."""
        try:
            # macOS의 기본 한글 폰트 : "/System/Library/Fonts/AppleSDGothicNeo.ttc"
            font_path = "/Users/localgroup/Library/Fonts/NanumSquareOTF_acEB.otf"
            if not os.path.exists(font_path):
                font_path = None
        except:
            font_path = None
        
        # 이진 탐색으로 최적 폰트 크기 찾기
        left, right = min_size, max_size
        optimal_size = min_size
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                if font_path:
                    test_font = ImageFont.truetype(font_path, mid)
                else:
                    test_font = ImageFont.load_default()
            except:
                test_font = ImageFont.load_default()
            
            if single_line:
                # 한 줄 텍스트의 경우
                bbox = draw.textbbox((0, 0), text, font=test_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                if text_width <= max_width and text_height <= max_height:
                    optimal_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                # 여러 줄 텍스트의 경우 (간단한 추정)
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=test_font)
                    text_width = bbox[2] - bbox[0]
                    
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            current_line = word
                            
                if current_line:
                    lines.append(current_line)
                
                # 전체 높이 계산
                bbox = draw.textbbox((0, 0), "A", font=test_font)
                line_height = bbox[3] - bbox[1] + 2
                total_height = len(lines) * line_height
                
                if total_height <= max_height:
                    optimal_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
        
        try:
            if font_path:
                return ImageFont.truetype(font_path, optimal_size)
            else:
                return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def _draw_text_in_box(self, draw, text, x, y, max_width, max_height, font, color):
        """박스 안에 한 줄 텍스트를 그립니다."""
        # 최적 폰트 크기 찾기
        optimal_font = self._find_optimal_font_size(draw, text, max_width, max_height, single_line=True)
        
        # 텍스트가 여전히 너무 길면 줄임표 추가
        display_text = text
        while len(display_text) > 0:
            bbox = draw.textbbox((0, 0), display_text, font=optimal_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if text_width <= max_width and text_height <= max_height:
                break
            display_text = display_text[:-1]
        
        if len(display_text) < len(text):
            # 줄임표를 추가하되 다시 크기 체크
            while len(display_text) > 0:
                test_text = display_text + "..."
                bbox = draw.textbbox((0, 0), test_text, font=optimal_font)
                if bbox[2] - bbox[0] <= max_width:
                    display_text = test_text
                    break
                display_text = display_text[:-1]
            
        # 텍스트를 수직 중앙에 배치
        bbox = draw.textbbox((0, 0), display_text, font=optimal_font)
        text_height = bbox[3] - bbox[1]
        center_y = int(y + (max_height - text_height) // 2)
        
        draw.text((x, center_y), display_text, fill=color, font=optimal_font)
    
    def _draw_multiline_text_in_box(self, draw, text, x, y, max_width, max_height, font, color):
        """박스 안에 여러 줄 텍스트를 그립니다."""
        # 최적 폰트 크기 찾기
        optimal_font = self._find_optimal_font_size(draw, text, max_width, max_height, single_line=False)
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=optimal_font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # 단어가 너무 길면 잘라서 사용
                    current_line = word[:int(max_width)//10]
                    
        if current_line:
            lines.append(current_line)
        
        # 줄 높이 계산
        bbox = draw.textbbox((0, 0), "A", font=optimal_font)
        line_height = bbox[3] - bbox[1] + 2
        
        # 최대 줄 수 계산
        max_lines = int(max_height // line_height)
        lines = lines[:max_lines]
        
        # 전체 텍스트 높이 계산
        total_text_height = len(lines) * line_height
        
        # 수직 중앙 정렬
        start_y = int(y + (max_height - total_text_height) // 2)
        
        # 텍스트 그리기
        for i, line in enumerate(lines):
            line_y = start_y + i * line_height
            if line_y + line_height <= y + max_height:
                draw.text((x, line_y), line, fill=color, font=optimal_font)

    def draw_layout(self, labels: torch.Tensor, bboxes: torch.Tensor):
        _canvas_width = self.canvas_width * self.times
        _canvas_height = self.canvas_height * self.times
        img = Image.new("RGB", (int(_canvas_width), int(_canvas_height)), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        labels_list = labels.tolist()
        bboxes_list = bboxes.tolist()
        areas = [bbox[2] * bbox[3] for bbox in bboxes_list]
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        for i in indices:
            bbox, label = bboxes_list[i], labels_list[i]
            color = self.colors[label]
            c_fill = color + (100,)
            x1, y1, x2, y2 = bbox
            x2 += x1
            y2 += y1
            x1, x2 = x1 * _canvas_width, x2 * _canvas_width
            y1, y2 = y1 * _canvas_height, y2 * _canvas_height
            draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
        return img

    @property
    def colors(self):
        if self._colors is None:
            n_colors = len(ID2LABEL[self.dataset]) + 1
            colors = sns.color_palette("husl", n_colors=n_colors)
            self._colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
        return self._colors

    def __call__(self, predictions, copy=None, show_bbox=True):
        images = []
        for i, prediction in enumerate(predictions):
            labels, bboxes = prediction
            
            # copy가 제공되면 해당 인덱스의 콘텐츠 데이터를 사용
            if copy and i < len(copy):
                img = self.draw_layout_with_content(labels, bboxes, copy[i], show_bbox=show_bbox)
            else:
                img = self.draw_layout(labels, bboxes)
            images.append(img)
        return images


class ContentAwareVisualizer:
    def __init__(self, times: float = 3):
        self.canvas_path = os.path.join(
            RAW_DATA_PATH("posterlayout"), "./test/image_canvas"
        )
        self.canvas_width, self.canvas_height = CANVAS_SIZE["posterlayout"]
        self.canvas_width *= times
        self.canvas_height *= times

    def draw_layout(self, img, elems, elems2):
        drawn_outline = img.copy()
        drawn_fill = img.copy()
        draw_ol = ImageDraw.ImageDraw(drawn_outline)
        draw_f = ImageDraw.ImageDraw(drawn_fill)
        cls_color_dict = {1: "green", 2: "red", 3: "orange"}

        for cls, box in elems:
            if cls[0]:
                draw_ol.rectangle(
                    tuple(box), fill=None, outline=cls_color_dict[cls[0]], width=5
                )

        s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
        for cls, box in s_elems:
            if cls[0]:
                draw_f.rectangle(tuple(box), fill=cls_color_dict[cls[0]])

        drawn_outline = drawn_outline.convert("RGBA")
        drawn_fill = drawn_fill.convert("RGBA")
        drawn_fill.putalpha(int(256 * 0.3))
        drawn = Image.alpha_composite(drawn_outline, drawn_fill)

        return drawn

    def __call__(self, predictions, test_idx):
        images = []
        pic = (
            Image.open(os.path.join(self.canvas_path, f"{test_idx}.png"))
            .convert("RGB")
            .resize((int(self.canvas_width), int(self.canvas_height)))
        )
        for prediction in predictions:
            labels, bboxes = prediction
            labels = labels.unsqueeze(-1)
            labels = np.array(labels, dtype=int)
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] *= self.canvas_width
            bboxes[:, 1::2] *= self.canvas_height
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            images.append(
                self.draw_layout(pic, zip(labels, bboxes), zip(labels, bboxes))
            )
        return images


def create_image_grid(
    image_list, rows=2, cols=5, border_size=6, border_color=(0, 0, 0)
):
    result_width = (
        image_list[0].width * cols + (cols - 1) * border_size + 2 * border_size
    )
    result_height = (
        image_list[0].height * rows + (rows - 1) * border_size + 2 * border_size
    )
    result_image = Image.new("RGB", (result_width, result_height), border_color)
    draw = ImageDraw.Draw(result_image)

    outer_border_rect = [0, 0, result_width, result_height]
    draw.rectangle(outer_border_rect, outline=border_color, width=border_size)

    for i in range(len(image_list)):
        row = i // cols
        col = i % cols
        x_offset = col * (image_list[i].width + border_size) + border_size
        y_offset = row * (image_list[i].height + border_size) + border_size
        result_image.paste(image_list[i], (x_offset, y_offset))

        if border_size > 0:
            border_rect = [
                x_offset - border_size,
                y_offset - border_size,
                x_offset + image_list[i].width + border_size,
                y_offset + image_list[i].height + border_size,
            ]
            draw.rectangle(border_rect, outline=border_color, width=border_size)

    return result_image