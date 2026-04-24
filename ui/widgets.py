from __future__ import annotations

from typing import Iterable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ui.style import CARD, INK, PAPER, SAGE, SLATE, TERRACOTTA


def repolish(widget: QtWidgets.QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def build_alignment_logo_pixmap(size: int = 88) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    outer_rect = QtCore.QRectF(4, 4, size - 8, size - 8)
    inner_rect = outer_rect.adjusted(10, 10, -10, -10)
    center = outer_rect.center()

    painter.setPen(QtGui.QPen(QtGui.QColor(INK), 4))
    painter.setBrush(QtGui.QColor(CARD))
    painter.drawEllipse(outer_rect)

    painter.setPen(QtGui.QPen(QtGui.QColor(TERRACOTTA), 8, cap=QtCore.Qt.RoundCap))
    path_a = QtGui.QPainterPath()
    path_a.moveTo(size * 0.20, size * 0.64)
    path_a.cubicTo(size * 0.34, size * 0.28, size * 0.55, size * 0.27, size * 0.78, size * 0.42)
    painter.drawPath(path_a)

    painter.setPen(QtGui.QPen(QtGui.QColor(SLATE), 8, cap=QtCore.Qt.RoundCap))
    path_b = QtGui.QPainterPath()
    path_b.moveTo(size * 0.22, size * 0.40)
    path_b.cubicTo(size * 0.45, size * 0.55, size * 0.63, size * 0.75, size * 0.80, size * 0.60)
    painter.drawPath(path_b)

    painter.setPen(QtGui.QPen(QtGui.QColor(SAGE), 4, cap=QtCore.Qt.RoundCap))
    painter.drawEllipse(inner_rect)
    painter.drawLine(QtCore.QPointF(size * 0.32, center.y()), QtCore.QPointF(size * 0.68, center.y()))
    painter.drawLine(QtCore.QPointF(center.x(), size * 0.32), QtCore.QPointF(center.x(), size * 0.68))

    painter.setBrush(QtGui.QColor(INK))
    painter.setPen(QtCore.Qt.NoPen)
    node_radius = max(4.0, size * 0.045)
    for x_ratio, y_ratio in ((0.26, 0.26), (0.74, 0.26), (0.26, 0.74), (0.74, 0.74)):
        painter.drawEllipse(
            QtCore.QRectF(
                size * x_ratio - node_radius,
                size * y_ratio - node_radius,
                node_radius * 2,
                node_radius * 2,
            )
        )

    painter.setBrush(QtGui.QColor("#FFF8F1"))
    painter.drawEllipse(QtCore.QRectF(center.x() - 6, center.y() - 6, 12, 12))
    painter.setBrush(QtGui.QColor(INK))
    painter.drawEllipse(QtCore.QRectF(center.x() - 3, center.y() - 3, 6, 6))
    painter.end()
    return pixmap


class TechPanel(QtWidgets.QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QtGui.QColor(PAPER))
        gradient.setColorAt(0.52, QtGui.QColor(CARD))
        gradient.setColorAt(1.0, QtGui.QColor("#E7D4C3"))
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), 28, 28)
        painter.fillPath(path, QtGui.QBrush(gradient))

        painter.setClipPath(path)
        painter.setPen(QtGui.QPen(QtGui.QColor(94, 80, 69, 24), 1))
        step = 34
        for x in range(rect.left(), rect.right(), step):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(rect.top(), rect.bottom(), step):
            painter.drawLine(rect.left(), y, rect.right(), y)


class PreviewTile(TechPanel):
    double_clicked = QtCore.pyqtSignal()

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._source_pixmap = QtGui.QPixmap()
        self._identity_text = title or "未加载"
        self._meta_text = "暂无数据"
        self._status_text = "等待数据"

        self.title_label = QtWidgets.QLabel(self._identity_text)
        self.title_label.setProperty("previewTitle", True)

        self.status_label = QtWidgets.QLabel(self._status_text)
        self.status_label.setProperty("chipRole", "status")
        self.status_label.setProperty("level", "warning")

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.status_label, 0, QtCore.Qt.AlignRight)

        self.image_label = QtWidgets.QLabel("等待数据")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_label.setProperty("previewCanvas", True)

        self.meta_label = QtWidgets.QLabel(self._meta_text)
        self.meta_label.setProperty("previewMeta", True)
        self.meta_label.setWordWrap(True)
        self.meta_label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.meta_label)
        self._refresh_tooltip()

    def set_identity(self, text: str) -> None:
        self._identity_text = text or "未加载"
        self.title_label.setText(self._identity_text)
        self._refresh_tooltip()

    def set_meta(self, text: str) -> None:
        self._meta_text = text or "暂无数据"
        self.meta_label.setText(self._meta_text)
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        lines = [self._identity_text, self._meta_text]
        if self._status_text:
            lines.append(f"状态：{self._status_text}")
        self.image_label.setToolTip("\n".join(lines))

    def set_status(self, text: str, level: str = "normal") -> None:
        self._status_text = text or "未加载"
        self.status_label.setText(self._status_text)
        self.status_label.setProperty("level", level)
        repolish(self.status_label)
        self._refresh_tooltip()

    def set_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            self._source_pixmap = QtGui.QPixmap()
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.setText("等待数据")
            return
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        self.image_label.setText("")
        self.image_label.setPixmap(
            self._source_pixmap.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()


class DraggableOverlayLabel(QtWidgets.QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source_pixmap = QtGui.QPixmap()
        self._drag_offset: QtCore.QPoint | None = None
        self._min_side = 180
        self._max_side = 16384
        self.setProperty("skeletonOverlay", True)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setText("骨架未生成")
        self.setToolTip("拖拽移动骨架叠加层，滚轮缩放，双击重置位置。")
        self.hide()

    def _max_allowed_side(self) -> int:
        return self._max_side

    def set_overlay_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        self._source_pixmap = pixmap or QtGui.QPixmap()
        self._refresh_pixmap()

    def reset_in_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        side = max(self._min_side, int(min(parent.width(), parent.height()) * 0.42))
        self.resize(side, side)
        self.move(max(18, parent.width() - self.width() - 22), 18)
        self._refresh_pixmap()
        self.clamp_to_parent()

    def clamp_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        min_x = min(0, parent.width() - self.width())
        max_x = max(0, parent.width() - self.width())
        min_y = min(0, parent.height() - self.height())
        max_y = max(0, parent.height() - self.height())
        self.move(max(min_x, min(self.x(), max_x)), max(min_y, min(self.y(), max_y)))

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            self.setPixmap(QtGui.QPixmap())
            self.setText("骨架未生成")
            return
        target = self.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setText("")
        self.setPixmap(
            self._source_pixmap.scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and (event.buttons() & QtCore.Qt.LeftButton):
            self.move(self.pos() + event.pos() - self._drag_offset)
            self.clamp_to_parent()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_in_parent()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        angle = event.angleDelta().y()
        if angle == 0:
            super().wheelEvent(event)
            return
        factor = 1.08 if angle > 0 else 0.92
        next_side = int(round(self.width() * factor))
        next_side = max(self._min_side, min(self._max_allowed_side(), next_side))
        center = self.geometry().center()
        self.resize(next_side, next_side)
        self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)
        self.clamp_to_parent()
        self._refresh_pixmap()
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()


class InfoRow(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setProperty("metricTitle", True)

        self.value_frame = QtWidgets.QFrame()
        self.value_frame.setProperty("infoValueCard", True)
        value_layout = QtWidgets.QVBoxLayout(self.value_frame)
        value_layout.setContentsMargins(14, 12, 14, 12)
        value_layout.setSpacing(0)

        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setProperty("metricValue", True)
        self.value_label.setWordWrap(True)
        self.value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value_layout.addWidget(self.value_label)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_frame)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


def rgb_array_to_qpixmap(frame: np.ndarray) -> QtGui.QPixmap:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("图像数组必须是 HxWx3 的 RGB 格式。")
    image = np.ascontiguousarray(frame[:, :, :3])
    qimage = QtGui.QImage(
        image.data,
        image.shape[1],
        image.shape[0],
        image.strides[0],
        QtGui.QImage.Format_RGB888,
    ).copy()
    return QtGui.QPixmap.fromImage(qimage)


def selected_checkboxes(checkboxes: Iterable[QtWidgets.QCheckBox]) -> tuple[str, ...]:
    return tuple(box.text() for box in checkboxes if box.isChecked())
