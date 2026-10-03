"""PySide6 status visualizations and cross-thread signals."""

import logging
import math

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from .config import UI_STYLES, WINDOW_SIZE


LOGGER = logging.getLogger("voice_client")

ORB_STATES = {
    "IDLE": ((110, 120, 145), 36, 0.020),
    "LISTENING": ((70, 175, 255), 41, 0.040),
    "HEARING": ((255, 165, 65), 45, 0.085),
    "THINKING": ((155, 95, 255), 43, 0.075),
    "SPEAKING": ((70, 245, 180), 46, 0.120),
    "ERROR": ((255, 70, 85), 44, 0.030),
}

PILL_WIDTH = 280
PILL_HEIGHT = 72


class AssistantSignals(QObject):
    state_changed = Signal(str)
    audio_level_changed = Signal(float)
    shutdown_requested = Signal()


class Orb(QWidget):
    def __init__(self, signals, stop_event, style="orb"):
        super().__init__()
        if style not in UI_STYLES:
            raise ValueError(f"Unsupported UI style: {style}")

        self.signals = signals
        self.stop_event = stop_event
        self.style = style
        self.state = "IDLE"
        self.phase = 0.0
        self.audio_level = 0.0
        self.drag_offset = None

        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        if style == "spectrum-pill":
            self.setFixedSize(PILL_WIDTH, PILL_HEIGHT)
        else:
            self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)

        self.signals.state_changed.connect(self.set_state)
        self.signals.audio_level_changed.connect(self.set_audio_level)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def set_state(self, state):
        if state in ORB_STATES:
            self.state = state
            self.update()

    def animate(self):
        self.phase += ORB_STATES[self.state][2]
        self.audio_level *= 0.92
        self.update()

    def set_audio_level(self, level):
        level = max(0.0, min(float(level), 1.0))
        self.audio_level = max(level, self.audio_level * 0.65)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        rgb, base_radius, _speed = ORB_STATES[self.state]
        color = QColor(*rgb)

        if self.style == "spectrum-pill":
            self._paint_spectrum_pill(painter, color, rgb)
        elif self.style == "circular-wave":
            self._paint_circular_wave(painter, cx, cy, color, rgb)
        else:
            self._paint_orb(painter, cx, cy, color, rgb, base_radius)

    def _paint_orb(self, painter, cx, cy, color, rgb, base_radius):
        pulse = (math.sin(self.phase) + 1.0) / 2.0
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

    def _paint_circular_wave(self, painter, cx, cy, color, rgb):
        pulse = (math.sin(self.phase * 1.4) + 1.0) / 2.0
        core_radius = 22 + pulse * 4

        ring_count = 1 if self.state == "IDLE" else 3
        for index in range(ring_count):
            progress = (
                self.phase / (math.pi * 2)
                + index / ring_count
            ) % 1.0

            if self.state == "THINKING":
                progress = 1.0 - progress

            radius = core_radius + 12 + progress * 62
            alpha = int((1.0 - progress) ** 1.5 * 175)
            if self.state == "IDLE":
                alpha = min(alpha, 70)

            width = 1.5 + (1.0 - progress) * 2.5
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(*rgb, alpha), width))
            painter.drawEllipse(
                int(cx - radius),
                int(cy - radius),
                int(radius * 2),
                int(radius * 2),
            )

        glow_radius = core_radius * 2.1
        glow = QRadialGradient(cx, cy, glow_radius)
        glow.setColorAt(0.0, QColor(*rgb, 220))
        glow.setColorAt(0.45, QColor(*rgb, 110))
        glow.setColorAt(1.0, QColor(*rgb, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(
            int(cx - glow_radius),
            int(cy - glow_radius),
            int(glow_radius * 2),
            int(glow_radius * 2),
        )

        core = QRadialGradient(
            cx - core_radius * 0.25,
            cy - core_radius * 0.30,
            core_radius * 1.4,
        )
        core.setColorAt(0.0, QColor(255, 255, 255, 245))
        core.setColorAt(
            0.30,
            QColor(
                min(color.red() + 75, 255),
                min(color.green() + 75, 255),
                min(color.blue() + 75, 255),
                245,
            ),
        )
        core.setColorAt(1.0, color)
        painter.setBrush(core)
        painter.drawEllipse(
            int(cx - core_radius),
            int(cy - core_radius),
            int(core_radius * 2),
            int(core_radius * 2),
        )

    def _paint_spectrum_pill(self, painter, color, rgb):
        painter.setPen(QPen(QColor(*rgb, 90), 1.5))
        painter.setBrush(QColor(18, 22, 31, 225))
        painter.drawRoundedRect(
            1,
            1,
            self.width() - 2,
            self.height() - 2,
            22,
            22,
        )

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(*rgb, 225))
        painter.drawEllipse(18, 28, 16, 16)

        bar_count = 15
        bar_width = 5
        gap = 4
        start_x = 48
        center_y = self.height() / 2

        state_levels = {
            "IDLE": 0.10,
            "LISTENING": 0.22,
            "HEARING": max(0.12, self.audio_level),
            "THINKING": 0.48,
            "SPEAKING": 0.72,
            "ERROR": 0.12,
        }
        level = state_levels[self.state]

        for index in range(bar_count):
            position = index / max(1, bar_count - 1)
            envelope = math.sin(position * math.pi) * 0.65 + 0.35
            motion = (
                math.sin(self.phase * 2.6 + index * 0.78) + 1.0
            ) / 2.0

            if self.state == "ERROR":
                motion = 0.08
            elif self.state == "HEARING":
                motion = max(motion * 0.65, self.audio_level)

            height = 4 + level * envelope * (12 + motion * 24)
            x = start_x + index * (bar_width + gap)
            painter.setBrush(QColor(*rgb, 155 + int(motion * 90)))
            painter.drawRoundedRect(
                int(x),
                int(center_y - height / 2),
                bar_width,
                int(height),
                2.5,
                2.5,
            )

        labels = {
            "IDLE": "Idle",
            "LISTENING": "Listening",
            "HEARING": "Hearing",
            "THINKING": "Thinking",
            "SPEAKING": "Speaking",
            "ERROR": "Error",
        }
        painter.setPen(QColor(235, 239, 247, 230))
        painter.drawText(
            190,
            0,
            self.width() - 206,
            self.height(),
            Qt.AlignVCenter | Qt.AlignLeft,
            labels[self.state],
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                self.drag_offset = None
                return

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
        LOGGER.info("Status window closed; stopping voice assistant")
        self.stop_event.set()
        self.signals.shutdown_requested.emit()
        event.accept()
