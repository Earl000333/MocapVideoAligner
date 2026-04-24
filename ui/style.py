from __future__ import annotations

from PyQt5 import QtGui


TERRACOTTA = "#AF5A3B"
SAGE = "#667C58"
SLATE = "#355166"
INK = "#241C17"
MUTED = "#5E5045"
PAPER = "#E8D9C8"
CARD = "#F3E6D7"
CARD_ALT = "#EAD9CA"
CANVAS = "#17120F"
ROSE = "#A94F5B"


def create_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(PAPER))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(INK))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(CARD))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(CARD_ALT))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(CARD))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(INK))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(INK))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(CARD))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(INK))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(TERRACOTTA))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#FFF8F1"))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(SLATE))
    return palette


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {PAPER};
        color: {INK};
        font-family: "Microsoft YaHei", "Segoe UI";
        font-size: 17px;
    }}
    QMainWindow, QDialog {{
        background-color: {PAPER};
    }}
    QLabel {{
        background: transparent;
        color: {INK};
    }}
    QLabel[role="title"] {{
        font-size: 38px;
        font-weight: 900;
        color: {INK};
        letter-spacing: 1px;
    }}
    QLabel[role="subtitle"], QLabel[muted="true"], QLabel[helpText="true"] {{
        color: {MUTED};
        font-size: 18px;
    }}
    QLabel[headerValue="true"] {{
        background: rgba(36, 28, 23, 0.06);
        border: 2px solid rgba(53, 81, 102, 0.16);
        border-radius: 20px;
        padding: 12px 22px;
        font-size: 20px;
        font-weight: 800;
        color: {INK};
    }}
    QLabel[panelTitle="true"] {{
        font-size: 28px;
        font-weight: 900;
        color: {INK};
    }}
    QLabel[panelAccent="true"] {{
        color: {TERRACOTTA};
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 2px;
    }}
    QLabel[stateBadge="true"] {{
        border-radius: 22px;
        padding: 10px 22px;
        font-size: 18px;
        font-weight: 900;
        min-width: 168px;
        color: #fff8f2;
        background: {TERRACOTTA};
        border: 2px solid rgba(175, 90, 59, 0.35);
    }}
    QLabel[stateBadge="true"][level="normal"] {{
        background: {SAGE};
        border-color: rgba(102, 124, 88, 0.42);
    }}
    QLabel[stateBadge="true"][level="warning"] {{
        background: {TERRACOTTA};
        border-color: rgba(175, 90, 59, 0.42);
    }}
    QLabel[stateBadge="true"][level="danger"] {{
        background: {ROSE};
        border-color: rgba(169, 79, 91, 0.42);
    }}
    QLabel[chipRole="status"] {{
        border-radius: 18px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 900;
        background: rgba(175, 90, 59, 0.12);
        border: 2px solid rgba(175, 90, 59, 0.22);
        color: {TERRACOTTA};
    }}
    QLabel[chipRole="status"][level="normal"] {{
        background: rgba(102, 124, 88, 0.16);
        border-color: rgba(102, 124, 88, 0.28);
        color: {SAGE};
    }}
    QLabel[chipRole="status"][level="warning"] {{
        background: rgba(175, 90, 59, 0.12);
        border-color: rgba(175, 90, 59, 0.24);
        color: {TERRACOTTA};
    }}
    QLabel[chipRole="status"][level="danger"] {{
        background: rgba(169, 79, 91, 0.12);
        border-color: rgba(169, 79, 91, 0.22);
        color: {ROSE};
    }}
    QLabel[previewTitle="true"] {{
        font-size: 22px;
        font-weight: 900;
        color: {INK};
    }}
    QLabel[previewMeta="true"] {{
        font-size: 30px;
        font-weight: 800;
        color: {MUTED};
    }}
    QLabel[previewCanvas="true"] {{
        border-radius: 24px;
        background-color: {CANVAS};
        border: 3px solid rgba(53, 81, 102, 0.14);
        color: #d8cfc6;
        font-size: 17px;
    }}
    QLabel[metricTitle="true"] {{
        color: {MUTED};
        font-weight: 800;
        font-size: 18px;
    }}
    QLabel[metricValue="true"] {{
        color: {INK};
        font-weight: 800;
        font-size: 18px;
    }}
    QFrame[card="true"] {{
        background-color: {CARD};
        border: 2px solid rgba(53, 81, 102, 0.12);
        border-radius: 28px;
    }}
    QFrame[infoValueCard="true"] {{
        background: rgba(255, 248, 241, 0.66);
        border: 2px solid rgba(53, 81, 102, 0.10);
        border-radius: 20px;
    }}
    QFrame[controlGroup="true"] {{
        background: rgba(255, 248, 241, 0.58);
        border: 2px solid rgba(53, 81, 102, 0.18);
        border-radius: 18px;
    }}
    QLabel[skeletonOverlay="true"] {{
        background: transparent;
        border: none;
        padding: 0px;
    }}
    QPushButton {{
        background-color: #f0dfcf;
        color: {INK};
        border: 2px solid rgba(53, 81, 102, 0.18);
        border-radius: 16px;
        padding: 10px 18px;
        font-size: 19px;
        font-weight: 900;
    }}
    QPushButton:hover {{
        background-color: #ebd6c3;
        border-color: rgba(53, 81, 102, 0.30);
    }}
    QPushButton:disabled {{
        color: #9a8b80;
        border-color: rgba(94, 80, 69, 0.12);
        background-color: #efe2d7;
    }}
    QPushButton[accent="true"] {{
        background-color: {TERRACOTTA};
        color: #fff8f2;
        border-color: rgba(175, 90, 59, 0.28);
    }}
    QPushButton[accent="true"]:hover {{
        background-color: #9f5236;
    }}
    QPushButton[danger="true"] {{
        background-color: {ROSE};
        color: #fff8f2;
        border-color: rgba(169, 79, 91, 0.28);
    }}
    QPushButton[danger="true"]:hover {{
        background-color: #984753;
    }}
    QPushButton[overlayToggle="true"] {{
        min-width: 176px;
        padding: 10px 16px;
        font-size: 18px;
    }}
    QPushButton[overlayToggle="true"]:checked {{
        background-color: {SLATE};
        color: #fff8f2;
        border-color: rgba(53, 81, 102, 0.30);
    }}
    QPushButton[overlayToggle="true"]:checked:hover {{
        background-color: #2f4759;
    }}
    QPushButton[playing="true"] {{
        background-color: {SLATE};
        color: #fff8f2;
        border-color: rgba(53, 81, 102, 0.30);
    }}
    QPushButton[playing="true"]:hover {{
        background-color: #2f4759;
    }}
    QLineEdit, QComboBox, QTextEdit, QPlainTextEdit {{
        background-color: #fbf3ea;
        color: {INK};
        border: 2px solid rgba(94, 80, 69, 0.18);
        border-radius: 18px;
        padding: 8px 12px;
        font-size: 16px;
        selection-background-color: {TERRACOTTA};
        selection-color: #fff8f2;
    }}
    QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 2px solid rgba(175, 90, 59, 0.56);
        background-color: #fff8f1;
    }}
    QComboBox QAbstractItemView {{
        background-color: #fbf3ea;
        color: {INK};
        selection-background-color: {TERRACOTTA};
        selection-color: #fff8f2;
        border: 2px solid rgba(94, 80, 69, 0.18);
    }}
    QComboBox[settingControl="true"] {{
        background-color: #fff6ec;
        border: 2px solid rgba(175, 90, 59, 0.24);
        border-radius: 16px;
        padding: 10px 14px;
        min-height: 26px;
        font-size: 18px;
        font-weight: 800;
    }}
    QComboBox[settingControl="true"]:focus {{
        border: 2px solid rgba(175, 90, 59, 0.52);
        background-color: #fff9f2;
    }}
    QCheckBox[settingToggle="true"] {{
        background: rgba(255, 248, 241, 0.62);
        border: 2px solid rgba(53, 81, 102, 0.14);
        border-radius: 16px;
        padding: 10px 14px;
        spacing: 10px;
        font-size: 18px;
        font-weight: 900;
    }}
    QCheckBox[settingToggle="true"]::indicator {{
        width: 22px;
        height: 22px;
    }}
    QGroupBox {{
        border: 2px solid rgba(53, 81, 102, 0.12);
        border-radius: 24px;
        margin-top: 22px;
        padding: 24px 14px 14px 14px;
        font-weight: 900;
        font-size: 20px;
        color: {INK};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 18px;
        padding: 0 10px;
        color: {TERRACOTTA};
    }}
    QPlainTextEdit[console="true"] {{
        background-color: #f8eee4;
        border: 2px solid rgba(53, 81, 102, 0.12);
        color: {INK};
        font-family: "Microsoft YaHei UI", "Microsoft YaHei";
        font-size: 16px;
        font-weight: 900;
    }}
    QSlider::groove:horizontal {{
        border: 0px;
        height: 20px;
        background: rgba(53, 81, 102, 0.12);
        border-radius: 10px;
    }}
    QSlider::sub-page:horizontal {{
        background: rgba(175, 90, 59, 0.76);
        border-radius: 10px;
    }}
    QSlider::add-page:horizontal {{
        background: rgba(53, 81, 102, 0.09);
        border-radius: 10px;
    }}
    QSlider::handle:horizontal {{
        background: {INK};
        width: 28px;
        margin: -8px 0;
        border-radius: 14px;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 16px;
        margin: 6px 0;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(53, 81, 102, 0.36);
        border-radius: 8px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
        height: 0px;
    }}
    QSplitter::handle {{
        background: rgba(94, 80, 69, 0.12);
    }}
    QSplitter::handle:hover {{
        background: rgba(94, 80, 69, 0.22);
    }}
    QCheckBox {{
        spacing: 10px;
        color: {INK};
        font-size: 18px;
        font-weight: 800;
    }}
    QCheckBox::indicator {{
        width: 22px;
        height: 22px;
    }}
    """
