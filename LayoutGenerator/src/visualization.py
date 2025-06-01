import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont

from .utilities import CANVAS_SIZE, ID2LABEL, RAW_DATA_PATH

# 상수 정의
class FontConfig:
    """폰트 설정 상수"""
    TITLE_FONT = "Paperlogy-9Black.ttf"
    DESCRIPTION_FONT = "Paperlogy-6SemiBold.ttf"
    TEXT_FONT = "Paperlogy-5Medium.ttf"
    
    TITLE_MIN_SIZE = 30
    TITLE_MAX_SIZE = 500
    TITLE_LINE_SPACING = 150
    
    DESCRIPTION_MIN_SIZE = 50
    DESCRIPTION_MAX_SIZE = 300
    DESCRIPTION_LINE_SPACING = 50
    
    TEXT_MIN_SIZE = 20
    TEXT_MAX_SIZE = 200
    TEXT_LINE_SPACING = 50

class LayoutConfig:
    """레이아웃 설정 상수"""
    TEXT_MARGIN = 2
    BBOX_ALPHA = 100
    DEFAULT_TEXT_COLOR = (0, 0, 0)
    ERROR_TEXT_COLOR = (128, 128, 128)
    BACKGROUND_COLOR = (255, 255, 255)


class Visualizer:
    def __init__(self, dataset: str, times: float = 3, image_filenames: list = []):
        self.dataset = dataset
        self.times = times
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]
        self._colors = None
        self._font_paths = self._initialize_font_paths()
        self.image_filenames = image_filenames
    
    def _initialize_font_paths(self) -> Dict[str, str]:
        """폰트 경로를 초기화합니다."""
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        return {
            'title': os.path.join(fonts_dir, FontConfig.TITLE_FONT),
            'description': os.path.join(fonts_dir, FontConfig.DESCRIPTION_FONT),
            'text': os.path.join(fonts_dir, FontConfig.TEXT_FONT)
        }
    
    def _get_canvas_size(self) -> Tuple[int, int]:
        """캔버스 크기를 반환합니다."""
        return (
            int(self.canvas_width * self.times),
            int(self.canvas_height * self.times)
        )
    
    def _create_canvas(self):
        """새로운 캔버스와 드로우 객체를 생성합니다."""
        canvas_width, canvas_height = self._get_canvas_size()
        img = Image.new("RGB", (canvas_width, canvas_height), color=LayoutConfig.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img, "RGBA")
        return img, draw

    def draw_layout_with_content(self, labels: torch.Tensor, bboxes: torch.Tensor, 
                                content_data: Dict[str, str], show_bbox: bool = True, image_index: int = 0) -> Image.Image:
        """
        레이아웃을 그리면서 바운딩 박스 안에 해당하는 텍스트 내용을 추가합니다.
        
        Args:
            labels: 레이블 텐서
            bboxes: 바운딩 박스 텐서
            content_data: {'title': '제목', 'description': '설명', 'image': '이미지', 'button': '버튼 텍스트'}
            show_bbox: 바운딩 박스를 보여줄지 여부 (기본값: True)
            image_index: 사용할 이미지의 인덱스 (기본값: 0)
        
        Returns:
            생성된 이미지
        """
        img, draw = self._create_canvas()
        canvas_width, canvas_height = self._get_canvas_size()
        
        # 레이아웃 요소들을 크기 순으로 정렬
        sorted_elements = self._sort_elements_by_area(labels, bboxes)
        id_to_label = ID2LABEL[self.dataset]
        
        # 각 레이블 타입별 카운터
        label_counters = {}
        
        for bbox, label in sorted_elements:
            # 박스 좌표 계산
            x1, y1, x2, y2 = self._calculate_box_coordinates(bbox, canvas_width, canvas_height)
            
            # 바운딩 박스 그리기
            if show_bbox:
                self._draw_bounding_box(draw, x1, y1, x2, y2, label)
            
            # 레이블명 가져오기
            label_name = id_to_label.get(label, f"label_{label}")
            
            # 레이블 타입별 카운터 업데이트
            if label_name not in label_counters:
                label_counters[label_name] = 0
            else:
                label_counters[label_name] += 1
            
            # 콘텐츠 렌더링
            self._render_content(img, draw, label_name, content_data, x1, y1, x2, y2, 
                               image_index, label_counters[label_name])
        
        return img
    
    def _sort_elements_by_area(self, labels: torch.Tensor, bboxes: torch.Tensor) -> List[Tuple[List, int]]:
        """요소들을 면적 순으로 정렬합니다."""
        labels_list = labels.tolist()
        bboxes_list = bboxes.tolist()
        areas = [bbox[2] * bbox[3] for bbox in bboxes_list]
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        return [(bboxes_list[i], labels_list[i]) for i in indices]
    
    def _calculate_box_coordinates(self, bbox: List[float], canvas_width: int, canvas_height: int) -> Tuple[float, float, float, float]:
        """박스 좌표를 계산합니다."""
        x1, y1, x2, y2 = bbox
        x2 += x1
        y2 += y1
        x1, x2 = x1 * canvas_width, x2 * canvas_width
        y1, y2 = y1 * canvas_height, y2 * canvas_height
        return x1, y1, x2, y2
    
    def _draw_bounding_box(self, draw, x1: float, y1: float, x2: float, y2: float, label: int):
        """바운딩 박스를 그립니다."""
        color = self.colors[label]
        c_fill = color + (LayoutConfig.BBOX_ALPHA,)
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    
    def _render_content(self, img, draw, label_name: str, 
                       content_data: Dict[str, str], x1: float, y1: float, x2: float, y2: float, 
                       image_index: int = 0, element_index: int = 0):
        """레이블 타입에 따라 적절한 콘텐츠를 렌더링합니다."""
        if label_name in ['title', 'description', 'text']:
            self._render_text_content(draw, label_name, content_data, x1, y1, x2, y2, element_index)
        elif label_name == 'image':
            current_image_index = (image_index + element_index) % len(self.image_filenames) if self.image_filenames else 0
            self._render_image_content(img, draw, x1, y1, x2, y2, current_image_index)
    
    def _find_optimal_font_size(self, draw, text: str, max_width: float, max_height: float, 
                               font_path: Optional[str] = None, single_line: bool = True, 
                               min_size: int = 8, max_size: int = 1000, line_spacing: int = 5):
        """주어진 박스 크기에 맞는 최적의 폰트 크기를 찾습니다."""
        left, right = min_size, max_size
        optimal_size = min_size
        
        while left <= right:
            mid = (left + right) // 2
            test_font = self._load_font(font_path, mid)
            
            if single_line:
                if self._fits_single_line(draw, text, test_font, max_width, max_height):
                    optimal_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if self._fits_multiline(draw, text, test_font, max_width, max_height, line_spacing):
                    optimal_size = mid
                    left = mid + 1
                else:
                    right = mid - 1
        
        return self._load_font(font_path, optimal_size)
    
    def _load_font(self, font_path: Optional[str], size: int):
        """폰트를 로드합니다."""
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            else:
                return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def _fits_single_line(self, draw, text: str, font, 
                         max_width: float, max_height: float) -> bool:
        """텍스트가 한 줄로 박스에 맞는지 확인합니다."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        return text_width <= max_width and text_height <= max_height
    
    def _fits_multiline(self, draw, text: str, font, 
                       max_width: float, max_height: float, line_spacing: int) -> bool:
        """텍스트가 여러 줄로 박스에 맞는지 확인합니다."""
        lines = self._split_text_into_lines(draw, text, font, max_width)
        total_height = self._calculate_text_height(draw, lines, font, line_spacing)
        return total_height <= max_height
    
    def _split_text_into_lines(self, draw, text: str, font, max_width: float) -> List[str]:
        """텍스트를 여러 줄로 분할합니다."""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
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
        
        return lines
    
    def _calculate_text_height(self, draw, lines: List[str], font, line_spacing: int) -> float:
        """텍스트의 총 높이를 계산합니다."""
        if not lines:
            return 0
        
        bbox = draw.textbbox((0, 0), "A", font=font)
        line_height = bbox[3] - bbox[1] + line_spacing
        return len(lines) * line_height - line_spacing
    
    def _draw_text_in_box(self, draw, text: str, x: float, y: float, max_width: float, max_height: float, 
                         color: Tuple[int, int, int], font_path: Optional[str] = None, 
                         min_font_size: int = 8, max_font_size: int = 200, line_spacing: int = 5):
        """박스 안에 텍스트를 그립니다. 길면 줄바꿈 처리."""
        # 최적 폰트 크기 찾기
        optimal_font = self._find_optimal_font_size(
            draw, text, max_width, max_height, font_path, 
            single_line=False, min_size=min_font_size, max_size=max_font_size, line_spacing=line_spacing
        )
        
        # 텍스트를 여러 줄로 분할
        lines = self._split_text_into_lines(draw, text, optimal_font, max_width)
        lines = self._limit_lines_to_fit(draw, lines, optimal_font, max_height, line_spacing)
        
        # 텍스트 렌더링
        self._render_text_lines(draw, lines, optimal_font, x, y, max_height, line_spacing, color)
    
    def _limit_lines_to_fit(self, draw, lines: List[str], font, max_height: float, line_spacing: int) -> List[str]:
        """최대 높이에 맞도록 줄 수를 제한합니다."""
        bbox = draw.textbbox((0, 0), "A", font=font)
        line_height = bbox[3] - bbox[1] + line_spacing
        max_lines = int(max_height // line_height)
        return lines[:max_lines]
    
    def _render_text_lines(self, draw, lines: List[str], font, x: float, y: float, 
                          max_height: float, line_spacing: int, color: Tuple[int, int, int]):
        """텍스트 줄들을 렌더링합니다."""
        if not lines:
            return
        
        bbox = draw.textbbox((0, 0), "A", font=font)
        line_height = bbox[3] - bbox[1] + line_spacing
        total_text_height = len(lines) * line_height - line_spacing
        
        # 세로 중앙 정렬, 가로 좌측 정렬
        start_y = int(y + (max_height - total_text_height) // 2)
        
        for i, line in enumerate(lines):
            line_y = start_y + i * line_height
            if line_y + line_height - line_spacing <= y + max_height:
                draw.text((x, line_y), line, fill=color, font=font)

    def draw_layout(self, labels: torch.Tensor, bboxes: torch.Tensor) -> Image.Image:
        """레이아웃만 그립니다 (텍스트 없이)."""
        img, draw = self._create_canvas()
        canvas_width, canvas_height = self._get_canvas_size()
        
        sorted_elements = self._sort_elements_by_area(labels, bboxes)
        
        for bbox, label in sorted_elements:
            x1, y1, x2, y2 = self._calculate_box_coordinates(bbox, canvas_width, canvas_height)
            self._draw_bounding_box(draw, x1, y1, x2, y2, label)
        
        return img

    @property
    def colors(self):
        """색상 팔레트를 반환합니다."""
        if self._colors is None:
            self._colors = self._generate_color_palette()
        return self._colors
    
    def _generate_color_palette(self):
        """색상 팔레트를 생성합니다."""
        n_colors = len(ID2LABEL[self.dataset]) + 1
        colors = sns.color_palette("husl", n_colors=n_colors)
        return [tuple(map(lambda x: int(x * 255), c)) for c in colors]

    def __call__(self, predictions, copy: Optional[List[Dict[str, str]]] = None, 
                show_bbox: bool = True):
        """예측 결과들을 시각화합니다."""
        images = []
        for i, prediction in enumerate(predictions):
            labels, bboxes = prediction
            
            # copy가 제공되면 해당 인덱스의 콘텐츠 데이터를 사용
            if copy and i < len(copy):
                # 각 레이아웃마다 다른 이미지 사용 (인덱스 기반으로 순환)
                img = self.draw_layout_with_content(labels, bboxes, copy[i], show_bbox=show_bbox, image_index=i)
            else:
                img = self.draw_layout(labels, bboxes)
            images.append(img)
        return images

    def _get_text_config(self, label_name: str) -> Dict[str, Union[str, int]]:
        """텍스트 타입에 따른 설정을 반환합니다."""
        configs = {
            'title': {
                'font_path': self._font_paths['title'],
                'min_size': FontConfig.TITLE_MIN_SIZE,
                'max_size': FontConfig.TITLE_MAX_SIZE,
                'line_spacing': FontConfig.TITLE_LINE_SPACING
            },
            'description': {
                'font_path': self._font_paths['description'],
                'min_size': FontConfig.DESCRIPTION_MIN_SIZE,
                'max_size': FontConfig.DESCRIPTION_MAX_SIZE,
                'line_spacing': FontConfig.DESCRIPTION_LINE_SPACING
            },
            'text': {
                'font_path': self._font_paths['text'],
                'min_size': FontConfig.TEXT_MIN_SIZE,
                'max_size': FontConfig.TEXT_MAX_SIZE,
                'line_spacing': FontConfig.TEXT_LINE_SPACING
            }
        }
        return configs.get(label_name, configs['text'])
    
    def _render_text_content(self, draw, label_name: str, 
                           content_data: Dict[str, str], x1: float, y1: float, x2: float, y2: float, element_index: int = 0):
        """텍스트 콘텐츠를 렌더링합니다."""
        # 인덱스에 따른 키 생성
        if element_index == 0:
            key = label_name
        else:
            key = f"{label_name}_{element_index + 1}"
        
        text_content = content_data.get(key, "")
        if not text_content:
            # 기본 키로 폴백
            text_content = content_data.get(label_name, "")
            if not text_content:
                return
        
        # 텍스트 영역 계산
        text_x = x1 + LayoutConfig.TEXT_MARGIN
        text_y = y1 + LayoutConfig.TEXT_MARGIN
        max_width = x2 - x1 - (LayoutConfig.TEXT_MARGIN * 2)
        max_height = y2 - y1 - (LayoutConfig.TEXT_MARGIN * 2)
        
        # 텍스트 타입에 따른 설정 가져오기
        font_config = self._get_text_config(label_name)
        
        self._draw_text_in_box(
            draw, text_content, text_x, text_y, max_width, max_height,
            LayoutConfig.DEFAULT_TEXT_COLOR, str(font_config['font_path']),
            int(font_config['min_size']), int(font_config['max_size']), int(font_config['line_spacing'])
        )
    
    def _render_image_content(self, img, draw, 
                            x1: float, y1: float, x2: float, y2: float, image_index: int = 0):
        """이미지 콘텐츠를 렌더링합니다."""
        # 이미지 파일명 선택 (인덱스가 범위를 벗어나면 첫 번째 이미지 사용)
        if not self.image_filenames:
            self._render_error_text(draw, "[이미지 목록 없음]", x1, y1, x2, y2)
            return
            
        image_filename = self.image_filenames[image_index % len(self.image_filenames)]
        image_path = os.path.join(os.path.dirname(__file__), "images", image_filename)
        
        if not os.path.exists(image_path):
            self._render_error_text(draw, f"[이미지 없음: {image_filename}]", x1, y1, x2, y2)
            return
        
        try:
            source_img = Image.open(image_path)
            resized_img = self._resize_image_to_fit(source_img, x1, y1, x2, y2)
            paste_x, paste_y = self._calculate_center_position(resized_img, x1, y1, x2, y2)
            
            # 투명도 처리
            if resized_img.mode != 'RGBA':
                resized_img = resized_img.convert('RGBA')
            
            img.paste(resized_img, (paste_x, paste_y), resized_img)
            
        except Exception as e:
            self._render_error_text(draw, "[이미지 오류]", x1, y1, x2, y2)
    
    def _resize_image_to_fit(self, source_img, x1: float, y1: float, x2: float, y2: float):
        """이미지를 박스 크기에 맞게 리사이즈합니다."""
        box_width = int(x2 - x1)
        box_height = int(y2 - y1)
        
        img_ratio = source_img.width / source_img.height
        box_ratio = box_width / box_height
        
        if img_ratio > box_ratio:
            new_width = box_width
            new_height = int(box_width / img_ratio)
        else:
            new_height = box_height
            new_width = int(box_height * img_ratio)
        
        return source_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _calculate_center_position(self, resized_img, x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
        """이미지를 중앙에 배치할 위치를 계산합니다."""
        box_width = int(x2 - x1)
        box_height = int(y2 - y1)
        paste_x = int(x1 + (box_width - resized_img.width) / 2)
        paste_y = int(y1 + (box_height - resized_img.height) / 2)
        return paste_x, paste_y
    
    def _render_error_text(self, draw, error_text: str, x1: float, y1: float, x2: float, y2: float):
        """오류 텍스트를 렌더링합니다."""
        text_x = x1 + LayoutConfig.TEXT_MARGIN
        text_y = y1 + LayoutConfig.TEXT_MARGIN
        max_width = x2 - x1 - (LayoutConfig.TEXT_MARGIN * 2)
        max_height = y2 - y1 - (LayoutConfig.TEXT_MARGIN * 2)
        
        self._draw_text_in_box(
            draw, error_text, text_x, text_y, max_width, max_height,
            LayoutConfig.ERROR_TEXT_COLOR, self._font_paths['text'],
            20, 100, 10
        )

    def visualize(self, ranked: List, copy=None, show_bbox=True) -> None:
        """레이아웃 시각화 및 저장"""
        if not ranked:
            print("시각화할 레이아웃이 없습니다.")
            return
            
        images = self.__call__(ranked, copy, show_bbox)
        grid_img = create_image_grid(images)
        
        # 출력 디렉토리 생성 및 저장 (LayoutGenerator/output)
        layout_generator_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(layout_generator_dir, "output", "output_poster.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_img.save(output_path)
        print(f"레이아웃 이미지가 저장되었습니다: {output_path}")

def create_image_grid(image_list: List[Image.Image], rows: int = 2, cols: int = 5, 
                     border_size: int = 6, border_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """이미지들을 그리드 형태로 배열합니다."""
    if not image_list:
        raise ValueError("이미지 리스트가 비어있습니다.")
    
    # 그리드 크기 계산
    grid_width, grid_height = _calculate_grid_dimensions(image_list[0], rows, cols, border_size)
    
    # 결과 이미지 생성
    result_image = Image.new("RGB", (grid_width, grid_height), border_color)
    draw = ImageDraw.Draw(result_image)
    
    # 외곽 테두리 그리기
    _draw_outer_border(draw, grid_width, grid_height, border_size, border_color)
    
    # 각 이미지 배치
    for i, img in enumerate(image_list):
        if i >= rows * cols:
            break
        _place_image_in_grid(result_image, draw, img, i, rows, cols, border_size, border_color)
    
    return result_image

def _calculate_grid_dimensions(sample_image: Image.Image, rows: int, cols: int, border_size: int) -> Tuple[int, int]:
    """그리드의 전체 크기를 계산합니다."""
    result_width = (
        sample_image.width * cols + (cols - 1) * border_size + 2 * border_size
    )
    result_height = (
        sample_image.height * rows + (rows - 1) * border_size + 2 * border_size
    )
    return result_width, result_height

def _draw_outer_border(draw, width: int, height: int, border_size: int, border_color: Tuple[int, int, int]):
    """외곽 테두리를 그립니다."""
    outer_border_rect = [0, 0, width, height]
    draw.rectangle(outer_border_rect, outline=border_color, width=border_size)

def _place_image_in_grid(result_image: Image.Image, draw, img: Image.Image, 
                        index: int, rows: int, cols: int, border_size: int, border_color: Tuple[int, int, int]):
    """그리드에 이미지를 배치합니다."""
    row = index // cols
    col = index % cols
    x_offset = col * (img.width + border_size) + border_size
    y_offset = row * (img.height + border_size) + border_size
    
    result_image.paste(img, (x_offset, y_offset))
    
    if border_size > 0:
        border_rect = [
            x_offset - border_size,
            y_offset - border_size,
            x_offset + img.width + border_size,
            y_offset + img.height + border_size,
        ]
        draw.rectangle(border_rect, outline=border_color, width=border_size)