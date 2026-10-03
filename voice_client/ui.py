"""PySide6 status orb and cross-thread signals."""

import logging
import math

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

from .config import WINDOW_SIZE


LOGGER = logging.getLogger("voice_client")

ORB_STATES = {
    "IDLE": ((110, 120, 145), 36, 0.020),
    "LISTENING": ((70, 175, 255), 41, 0.040),
    "HEARING": ((255, 165, 65), 45, 0.085),
    "THINKING": ((155, 95, 255), 43, 0.075),
    "SPEAKING": ((70, 245, 180), 46, 0.120),
    "ERROR": ((255, 70, 85), 44, 0.030),
}


class AssistantSignals(QObject):
    state_changed = Signal(str)
    shutdown_requested = Signal()


class Orb(QWidget):
    def __init__(self, signals, stop_event):
        super().__init__()
        self.signals = signals
        self.stop_event = stop_event
        self.state = "IDLE"
        self.phase = 0.0
        self.drag_offset = None

        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)
        self.signals.state_changed.connect(self.set_state)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def set_state(self, state):
        if state in ORB_STATES:
            self.state = state
            self.update()

    def animate(self):
        self.phase += ORB_STATES[self.state][2]
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        pulse = (math.sin(self.phase) + 1.0) / 2.0
        rgb, base_radius, _speed = ORB_STATES[self.state]
        color = QColor(*rgb)
        radius = base_radius + pulse * 8

        glow_radius = radius * 2.0
        glow = QRadialGradient(cx, cy, glow_radius)
        glow.setColorAt(0.0, QColor(*rgb, 185))
        glow.setColorAt(0.35, QColor(*rgb, 105))
        glow.setColorAt(1.0, QColor(*rgb, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(
            int(cx - glow_radius),
            int(cy - glow_radius),
            int(glow_radius * 2),
            int(glow_radius * 2),
        )

        inner = QRadialGradient(
            cx - radius * 0.25,
            cy - radius * 0.30,
            radius * 1.35,
        )
        inner.setColorAt(0.0, QColor(255, 255, 255, 245))
        inner.setColorAt(
            0.20,
            QColor(
                min(color.red() + 70, 255),
                min(color.green() + 70, 255),
                min(color.blue() + 70, 255),
                245,
            ),
        )
        inner.setColorAt(0.70, color)
        inner.setColorAt(
            1.0,
            QColor(
                color.red() // 2,
                color.green() // 2,
                color.blue() // 2,
                245,
            ),
        )
        painter.setBrush(inner)
        painter.drawEllipse(
            int(cx - radius),
            int(cy - radius),
            int(radius * 2),
            int(radius * 2),
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self.drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_offset)

    def mouseReleaseEvent(self, event):
        del event
        self.drag_offset = None

    def closeEvent(self, event):
        LOGGER.info("Orb closed; stopping voice assistant")
        self.stop_event.set()
        self.signals.shutdown_requested.emit()
        event.accept()
