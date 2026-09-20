APP_STYLE = """
QMainWindow {
    background-color: #0b0e13;
}

QWidget {
    color: #e9edf5;
    font-family: "Segoe UI";
    font-size: 13px;
}

QFrame#TopBar {
    background-color: #11151c;
    border-bottom: 1px solid #252b35;
}

QLabel#AppTitle {
    font-size: 20px;
    font-weight: 700;
    color: #f5f7fb;
}

QLabel#AppSubtitle {
    color: #7f8999;
    font-size: 11px;
}

QLabel#SectionTitle {
    color: #8f9bad;
    font-size: 11px;
    font-weight: 700;
}

QFrame#Panel {
    background-color: #10141b;
    border: 1px solid #222833;
    border-radius: 8px;
}

QFrame#VideoFrame {
    background-color: #050609;
    border: 1px solid #272e39;
    border-radius: 8px;
}

QLabel#VideoPlaceholder {
    color: #596273;
    font-size: 16px;
}

QLabel#SourceBadge {
    background-color: #171c24;
    border: 1px solid #303846;
    border-radius: 5px;
    padding: 5px 9px;
    color: #cbd2dc;
}

QPushButton {
    background-color: #191f29;
    border: 1px solid #303846;
    border-radius: 6px;
    padding: 7px 12px;
}

QPushButton:hover {
    background-color: #222a36;
}

QPushButton:pressed {
    background-color: #151a22;
}

QPushButton#PrimaryButton {
    background-color: #2867d6;
    border: 1px solid #3977df;
    color: white;
    font-weight: 600;
}

QPushButton#DangerButton {
    color: #ff8585;
}

QPushButton:disabled {
    color: #555e6d;
    background-color: #12161c;
    border-color: #202630;
}

QListWidget {
    background-color: transparent;
    border: none;
    outline: none;
}

QListWidget::item {
    padding: 9px;
    border-radius: 5px;
}

QListWidget::item:selected {
    background-color: #202938;
}

QListWidget::item:hover {
    background-color: #181e27;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QSpinBox,
QDoubleSpinBox,
QComboBox {
    background-color: #171c24;
    border: 1px solid #303846;
    border-radius: 5px;
    padding: 5px;
    min-height: 24px;
}

QSlider::groove:horizontal {
    background-color: #282f3a;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background-color: #d4dae4;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}

QTableWidget {
    background-color: #0f1319;
    border: 1px solid #222833;
    gridline-color: #1d232c;
    border-radius: 6px;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #23304a;
}

QHeaderView::section {
    background-color: #151a22;
    color: #8f9bad;
    border: none;
    border-bottom: 1px solid #252c36;
    padding: 7px;
    font-weight: 600;
}

QScrollBar:vertical {
    background: transparent;
    width: 9px;
}

QScrollBar::handle:vertical {
    background: #323a47;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #424c5c;
}

/*
    The splitter handle is intentionally slightly visible.
    This helps the user understand that the panels are resizable.
*/
QSplitter::handle {
    background-color: #181e26;
}

QSplitter::handle:hover {
    background-color: #35425a;
}

QStatusBar {
    background-color: #0d1015;
    color: #778293;
    border-top: 1px solid #202630;
}
"""
